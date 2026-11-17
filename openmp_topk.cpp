// =============================================================================
//  openmp_topk.cpp  —  OPENMP (multi-core CPU)
//  Exact top-k cosine similarity search with two top-k strategies:
//
//    mode "hier"  : hierarchical reduction INSIDE each query
//                   (each thread selects a local top-k from its slice of N,
//                    then one thread merges T*k candidates -> global top-k)
//    mode "query" : one query per thread (outer parallelism, no merge needed)
//
//  Build : g++ -O3 -march=native -fopenmp -std=c++17 openmp_topk.cpp -o openmp_topk
//  Run   : ./openmp_topk  N  d  Q  k  [batch]  [hier|query]
//          OMP_NUM_THREADS=8 ./openmp_topk 200000 384 64 10 32 hier
//
//  Tip: run with OMP_NUM_THREADS=1 to separate SIMD gains from thread gains.
// =============================================================================
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// ---------- normalisation (rows are independent -> parallel for) ----------
static void normalize_rows(float *X, long long rows, int d) {
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < rows; ++r) {
        float *row = X + r * d;
        float s = 0.0f;
#pragma omp simd reduction(+ : s)
        for (int j = 0; j < d; ++j) s += row[j] * row[j];
        float inv = s > 0.0f ? 1.0f / std::sqrt(s) : 0.0f;
#pragma omp simd
        for (int j = 0; j < d; ++j) row[j] *= inv;
    }
}

static inline float dot(const float *a, const float *b, int d) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (int j = 0; j < d; ++j) s += a[j] * b[j];
    return s;
}

// ---------- top-k helpers ----------
struct Hit { float score; int idx; };
static inline bool better(const Hit &a, const Hit &b) {
    return a.score > b.score || (a.score == b.score && a.idx < b.idx);
}

// push one candidate into a size-k min-heap (front = worst kept)
static inline void heap_offer(std::vector<Hit> &heap, int k, Hit h) {
    if ((int)heap.size() < k) {
        heap.push_back(h);
        std::push_heap(heap.begin(), heap.end(), better);
    } else if (better(h, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), better);
        heap.back() = h;
        std::push_heap(heap.begin(), heap.end(), better);
    }
}

static void write_sorted(std::vector<Hit> &heap, int k, Hit *out) {
    std::sort_heap(heap.begin(), heap.end(), better);
    for (int j = 0; j < k; ++j) out[j] = j < (int)heap.size() ? heap[j] : Hit{-INFINITY, -1};
}

// Strategy A: hierarchical reduction within each query
//   level 1: each thread -> local top-k of its contiguous slice of N
//   level 2: single thread merges T*k candidates -> global top-k
static void topk_hierarchical(const float *scores, int nb, long long N, int k, Hit *out) {
    int T = omp_get_max_threads();
    std::vector<Hit> cand((size_t)T * k);
    std::vector<int> cnt(T);

#pragma omp parallel
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        std::vector<Hit> heap;
        heap.reserve(k);
        long long chunk = (N + nt - 1) / nt;
        long long lo = t * chunk, hi = std::min(N, lo + chunk);

        for (int b = 0; b < nb; ++b) {
            const float *row = scores + (size_t)b * N;

            heap.clear();                                   // level 1: local selection
            for (long long i = lo; i < hi; ++i) heap_offer(heap, k, Hit{row[i], (int)i});
            cnt[t] = (int)heap.size();
            std::copy(heap.begin(), heap.end(), cand.begin() + (size_t)t * k);

#pragma omp barrier
#pragma omp single                                          // level 2: global merge
            {
                std::vector<Hit> g;
                g.reserve(k);
                for (int u = 0; u < nt; ++u)
                    for (int j = 0; j < cnt[u]; ++j) heap_offer(g, k, cand[(size_t)u * k + j]);
                write_sorted(g, k, out + (size_t)b * k);
            }   // implicit barrier: cand[] is free to reuse for the next query
        }
    }
}

// Strategy B: each thread handles whole queries
static void topk_per_query(const float *scores, int nb, long long N, int k, Hit *out) {
#pragma omp parallel
    {
        std::vector<Hit> heap;
        heap.reserve(k);
#pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < nb; ++b) {
            const float *row = scores + (size_t)b * N;
            heap.clear();
            for (long long i = 0; i < N; ++i) heap_offer(heap, k, Hit{row[i], (int)i});
            write_sorted(heap, k, out + (size_t)b * k);
        }
    }
}

int main(int argc, char **argv) {
    long long N = argc > 1 ? atoll(argv[1]) : 200000;
    int d       = argc > 2 ? atoi(argv[2])  : 384;
    int Q       = argc > 3 ? atoi(argv[3])  : 64;
    int k       = argc > 4 ? atoi(argv[4])  : 10;
    int batch   = argc > 5 ? atoi(argv[5])  : 32;
    const char *mode = argc > 6 ? argv[6] : "hier";
    bool hier = strcmp(mode, "query") != 0;
    if (k > N) k = (int)N;
    int T = omp_get_max_threads();

    printf("=== OPENMP (%s) | threads=%d | N=%lld d=%d Q=%d k=%d batch=%d ===\n",
           hier ? "hierarchical top-k" : "per-query top-k", T, N, d, Q, k, batch);

    std::vector<float> db((size_t)N * d), qs((size_t)Q * d);
    fill_random(db, 42);
    fill_random(qs, 7);

    double t0 = omp_get_wtime();
    normalize_rows(db.data(), N, d);
    normalize_rows(qs.data(), Q, d);
    double t_norm = omp_get_wtime() - t0, t_sim = 0.0, t_topk = 0.0;

    std::vector<float> scores((size_t)batch * N);
    std::vector<Hit> result((size_t)Q * k);

    for (int q0 = 0; q0 < Q; q0 += batch) {
        int nb = std::min(batch, Q - q0);

        // ---- similarity: split DATABASE rows across threads.
        //      Each db row is loaded once and reused for all nb queries (cache reuse).
        double a = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < N; ++i) {
            const float *x = db.data() + i * d;
            for (int b = 0; b < nb; ++b)
                scores[(size_t)b * N + i] = dot(qs.data() + (size_t)(q0 + b) * d, x, d);
        }
        double m = omp_get_wtime();

        // ---- top-k ----
        Hit *out = result.data() + (size_t)q0 * k;
        if (hier) topk_hierarchical(scores.data(), nb, N, k, out);
        else      topk_per_query(scores.data(), nb, N, k, out);
        double e = omp_get_wtime();

        t_sim += m - a;
        t_topk += e - m;
    }

    // ---- correctness check vs a plain serial reference (first few queries) ----
    int nverify = std::min(Q, 4);
    double recall = 0.0;
    std::vector<Hit> ref(N);
    for (int q = 0; q < nverify; ++q) {
        for (long long i = 0; i < N; ++i) ref[i] = Hit{dot(qs.data() + (size_t)q * d, db.data() + i * d, d), (int)i};
        std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), better);
        int hits = 0;
        for (int j = 0; j < k; ++j)
            for (int r = 0; r < k; ++r)
                if (result[(size_t)q * k + j].idx == ref[r].idx) { ++hits; break; }
        recall += (double)hits / k;
    }
    recall /= nverify;

    double total = t_norm + t_sim + t_topk;
    double gflops = 2.0 * Q * N * d / t_sim / 1e9;
    double checksum = 0.0;
    for (auto &h : result) checksum += h.score;

    printf("normalise   : %10.4f s\n", t_norm);
    printf("similarity  : %10.4f s   (%.2f GFLOP/s)\n", t_sim, gflops);
    printf("top-k       : %10.4f s   (%.1f%% of search time)\n", t_topk, 100.0 * t_topk / (t_sim + t_topk));
    printf("TOTAL       : %10.4f s\n", total);
    printf("throughput  : %10.1f queries/s\n", Q / (t_sim + t_topk));
    printf("recall@%d    : %.4f   (vs serial reference, first %d queries)\n", k, recall, nverify);
    printf("checksum    : %.6f   (sum of all top-k scores; compare across versions)\n", checksum);
    printf("query 0 top-%d:", std::min(k, 5));
    for (int j = 0; j < std::min(k, 5); ++j) printf("  [%d] %.4f", result[j].idx, result[j].score);
    printf("\n");
    printf("CSV,openmp-%s,%lld,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", hier ? "hier" : "query",
           N, d, Q, k, T, t_norm, t_sim, t_topk, total);
    return 0;
}
