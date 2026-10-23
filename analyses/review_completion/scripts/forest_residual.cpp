#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>
struct Forest {
    int d, e;
    std::vector<int> a, b, parent, pe, order, component, roots;
    std::vector<std::vector<std::pair<int, int>>> adj;
    Forest(int dd, int ee, const int *edges)
        : d(dd), e(ee), parent(d, -2), pe(d, -1), component(d, -1), adj(d) {
        for (int j = 0; j < e; j++) {
            a.push_back(edges[2 * j]);
            b.push_back(edges[2 * j + 1]);
            adj[a[j]].push_back({b[j], j});
            adj[b[j]].push_back({a[j], j});
        }
        for (int r = 0; r < d; r++)
            if (parent[r] == -2) {
                int cid = roots.size();
                roots.push_back(r);
                parent[r] = -1;
                component[r] = cid;
                order.push_back(r);
                for (size_t i = order.size() - 1; i < order.size(); i++) {
                    int v = order[i];
                    for (auto [u, edge] : adj[v])
                        if (parent[u] == -2) {
                            parent[u] = v;
                            pe[u] = edge;
                            component[u] = cid;
                            order.push_back(u);
                        }
                }
            }
    }
    void flows(const int *x, double *out) const {
        std::vector<double> d0(d, 0), d1(d, 0), u0(d, 0), u1(d, 0), base(roots.size(), 0);
        for (auto it = order.rbegin(); it != order.rend(); it++) {
            int v = *it;
            d1[v] = x[v];
            for (auto [u, j] : adj[v])
                if (parent[u] == v) {
                    d0[v] += d1[u];
                    d1[v] += std::min(d0[u], d1[u]);
                }
        }
        out[0] = 0;
        for (size_t c = 0; c < roots.size(); c++) {
            int r = roots[c];
            base[c] = std::min(d0[r], d1[r]);
            out[0] += base[c];
        }
        for (int v : order) {
            double t0 = d0[v] + u1[v], t1 = d1[v] + std::min(u0[v], u1[v]);
            for (auto [u, j] : adj[v])
                if (parent[u] == v) {
                    u0[u] = t0 - d1[u];
                    u1[u] = t1 - std::min(d0[u], d1[u]);
                    out[j + 1] = out[0] - base[component[v]] + std::min(d0[u], d1[u]) +
                                 std::min(u0[u], u1[u]);
                }
        }
    }
};
extern "C" void forest_values(const int *X, int n, int d, const int *edges, int e, double *out) {
    Forest f(d, e, edges);
    for (int i = 0; i < n; i++)
        f.flows(X + i * d, out + i * (e + 1));
}
extern "C" void forest_mc(const int64_t *X, const int64_t *L, int n, int d, const int *edges, int e,
                          int k, int batches, int draws, uint64_t seed, double *out) {
    Forest f(d, e, edges);
    int nf = e + 1;
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        int64_t S = 0;
        for (int g = 0; g < d; g++)
            S += X[i * d + g];
        double *o = out + i * batches * nf;
        std::fill(o, o + batches * nf, 0);
        int low = std::max<int64_t>(4, k - (L[i] - S)), high = std::min<int64_t>(k, S);
        if (low > high)
            continue;
        auto lc = [](int64_t nn, int64_t r) {
            return lgammal(nn + 1.L) - lgammal(r + 1.L) - lgammal(nn - r + 1.L);
        };
        int64_t support = 1;
        std::vector<int> active;
        for (int g = 0; g < d; g++)
            if (X[i * d + g] > 0) {
                active.push_back(g);
                support =
                    std::min<int64_t>(4097, support * (std::min<int64_t>(k, X[i * d + g]) + 1));
            }
        if (support <= 4096) {
            int minN = std::max<int64_t>(0, k - (L[i] - S)), maxN = std::min<int64_t>(k, S);
            std::vector<long double> wr(maxN + 1, 0);
            wr[minN] = expl(lc(L[i] - S, k - minN) - lc(L[i], k));
            for (int z = minN; z < maxN; z++)
                wr[z + 1] = wr[z] * (k - z) / (L[i] - S - k + z + 1.L);
            std::vector<int> counts(d, 0);
            std::vector<double> vals(nf);
            std::vector<long double> sum(nf, 0);
            std::function<void(int, int, long double)> rec = [&](int j, int n, long double w) {
                if (j == (int)active.size()) {
                    if (n < 4 || n < minN)
                        return;
                    f.flows(counts.data(), vals.data());
                    for (int h = 0; h < nf; h++)
                        sum[h] += w * wr[n] * vals[h] / k;
                    return;
                }
                int g = active[j];
                long double c = 1;
                for (int v = 0; v <= std::min<int64_t>(X[i * d + g], k - n); v++) {
                    counts[g] = v;
                    rec(j + 1, n + v, w * c);
                    c = c * (X[i * d + g] - v) / (v + 1.L);
                }
                counts[g] = 0;
            };
            rec(0, 0, 1);
            for (int b = 0; b < batches; b++)
                for (int h = 0; h < nf; h++)
                    o[b * nf + h] = (double)sum[h];
            continue;
        }
        std::vector<long double> cdf;
        long double w = expl(lc(S, low) + lc(L[i] - S, k - low) - lc(L[i], k)), tail = 0;
        for (int z = low; z <= high; z++) {
            tail += w;
            cdf.push_back(tail);
            if (z < high)
                w *= ((S - z) * (long double)(k - z)) / ((z + 1.L) * (L[i] - S - k + z + 1.L));
        }
        if (tail == 0)
            continue;
        std::mt19937_64 rng(seed + uint64_t(i) * 0x9e3779b97f4a7c15ULL +
                            uint64_t(k) * 0xbf58476d1ce4e5b9ULL);
        std::uniform_real_distribution<long double> unif(0, 1);
        std::vector<int> counts(d), remain(d);
        std::vector<double> flow(nf);
        for (int b = 0; b < batches; b++)
            for (int r = 0; r < draws; r++) {
                int z =
                    low + std::lower_bound(cdf.begin(), cdf.end(), unif(rng) * tail) - cdf.begin();
                std::fill(counts.begin(), counts.end(), 0);
                for (int g = 0; g < d; g++)
                    remain[g] = X[i * d + g];
                int64_t total = S;
                for (int t = 0; t < z; t++) {
                    std::uniform_int_distribution<int64_t> pick(0, total - 1);
                    int64_t u = pick(rng);
                    int g = 0;
                    while (u >= remain[g]) {
                        u -= remain[g];
                        g++;
                    }
                    counts[g]++;
                    remain[g]--;
                    total--;
                }
                f.flows(counts.data(), flow.data());
                for (int j = 0; j < nf; j++)
                    o[b * nf + j] += (double)tail * flow[j] / (k * draws);
            }
    }
}
