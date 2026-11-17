// =============================================================================
//  serial_topk.cpp  —  SERIAL BASELINE (brute force)
//  Exact top-k cosine similarity search, single thread.
//
//  Build : g++ -O3 -march=native -std=c++17 serial_topk.cpp -o serial_topk
//  Run   : ./serial_topk  N  d  Q  k  [batch]
//          e.g. ./serial_topk 200000 384 64 10
//
//  Pipeline (same for all three versions):
//    1. Generate database (N x d) and queries (Q x d) with a fixed-seed RNG
//    2. Normalise every row to unit length -> cosine sim == dot product
//    3. For each batch of queries: compute scores[batch][N]   (similarity)
//    4. For each query: select k best scores with a size-k min-heap (top-k)
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// ---------- deterministic RNG (identical in serial / OpenMP / CUDA) ----------
static inline uint64_t splitmix64(uint64_t &s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void fill_random(std::vector<float> &v, uint64_t seed) {
    uint64_t s = seed;
    for (auto &x : v) x = (float)((splitmix64(s) >> 40) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

// ---------- normalisation ----------
static void normalize_rows(float *X, long long rows, int d) {
    for (long long r = 0; r < rows; ++r) {
        float *row = X + r * d;
        float s = 0.0f;
        for (int j = 0; j < d; ++j) s += row[j] * row[j];
        float inv = s > 0.0f ? 1.0f / std::sqrt(s) : 0.0f;
        for (int j = 0; j < d; ++j) row[j] *= inv;
    }
}

static inline float dot(const float *a, const float *b, int d) {
    float s = 0.0f;
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

// ---------- top-k with a size-k min-heap ----------
struct Hit { float score; int idx; };
// "a is better than b": higher score wins, ties broken by smaller index
static inline bool better(const Hit &a, const Hit &b) {
    return a.score > b.score || (a.score == b.score && a.idx < b.idx);
}

// heap top (front) = WORST of the current k best
static void topk_row(const float *scores, int N, int k, Hit *out) {
    std::vector<Hit> heap;
    heap.reserve(k);
    for (int i = 0; i < N; ++i) {
        Hit h{scores[i], i};
        if ((int)heap.size() < k) {
            heap.push_back(h);
            std::push_heap(heap.begin(), heap.end(), better);
        } else if (better(h, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), better);
            heap.back() = h;
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }
    std::sort_heap(heap.begin(), heap.end(), better);   // best first
    for (int j = 0; j < k; ++j) out[j] = j < (int)heap.size() ? heap[j] : Hit{-INFINITY, -1};
}

int main(int argc, char **argv) {
    long long N = argc > 1 ? atoll(argv[1]) : 200000;
    int d       = argc > 2 ? atoi(argv[2])  : 384;
    int Q       = argc > 3 ? atoi(argv[3])  : 64;
    int k       = argc > 4 ? atoi(argv[4])  : 10;
    int batch   = argc > 5 ? atoi(argv[5])  : 32;
    if (k > N) k = (int)N;

    printf("=== SERIAL  |  N=%lld d=%d Q=%d k=%d batch=%d ===\n", N, d, Q, k, batch);

    std::vector<float> db((size_t)N * d), qs((size_t)Q * d);
    fill_random(db, 42);
    fill_random(qs, 7);

    auto t0 = Clock::now();
    normalize_rows(db.data(), N, d);
    normalize_rows(qs.data(), Q, d);
    auto t1 = Clock::now();
    double t_norm = secs(t0, t1), t_sim = 0.0, t_topk = 0.0;

    std::vector<float> scores((size_t)batch * N);
    std::vector<Hit> result((size_t)Q * k);

    for (int q0 = 0; q0 < Q; q0 += batch) {
        int nb = std::min(batch, Q - q0);

        // ---- similarity: scores[b][i] = <query b, db row i> ----
        auto a = Clock::now();
        for (long long i = 0; i < N; ++i) {
            const float *x = db.data() + i * d;
            for (int b = 0; b < nb; ++b)
                scores[(size_t)b * N + i] = dot(qs.data() + (size_t)(q0 + b) * d, x, d);
        }
        auto m = Clock::now();

        // ---- top-k selection ----
        for (int b = 0; b < nb; ++b)
            topk_row(scores.data() + (size_t)b * N, (int)N, k, result.data() + (size_t)(q0 + b) * k);
        auto e = Clock::now();

        t_sim += secs(a, m);
        t_topk += secs(m, e);
    }

    double total = t_norm + t_sim + t_topk;
    double gflops = 2.0 * Q * N * d / t_sim / 1e9;
    double checksum = 0.0;
    for (auto &h : result) checksum += h.score;

    printf("normalise   : %10.4f s\n", t_norm);
    printf("similarity  : %10.4f s   (%.2f GFLOP/s)\n", t_sim, gflops);
    printf("top-k       : %10.4f s   (%.1f%% of search time)\n", t_topk, 100.0 * t_topk / (t_sim + t_topk));
    printf("TOTAL       : %10.4f s\n", total);
    printf("throughput  : %10.1f queries/s\n", Q / (t_sim + t_topk));
    printf("checksum    : %.6f   (sum of all top-k scores; compare across versions)\n", checksum);
    printf("query 0 top-%d:", std::min(k, 5));
    for (int j = 0; j < std::min(k, 5); ++j) printf("  [%d] %.4f", result[j].idx, result[j].score);
    printf("\n");
    // machine-readable line for collecting results
    printf("CSV,serial,%lld,%d,%d,%d,1,%.6f,%.6f,%.6f,%.6f\n", N, d, Q, k, t_norm, t_sim, t_topk, total);
    return 0;
}
