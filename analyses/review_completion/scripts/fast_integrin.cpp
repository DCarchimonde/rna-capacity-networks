#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
// Sum out the fourth category analytically using prefix masses and moments.
// Exact for the finite hypergeometric support. Multinomial support is capped
// only when the caller supplies a documented occupancy cap.
static void e4(const double *q, const int64_t *x, int64_t L, int k, int cap, bool plugin,
               double *ans) {
    ans[0] = ans[1] = ans[2] = 0;
    if (plugin ? ((q[0] + q[1]) == 0 || (q[2] + q[3]) == 0)
               : ((x[0] + x[1]) == 0 || (x[2] + x[3]) == 0))
        return;
    int lim[4];
    int total = 0;
    double rho = 0;
    for (int j = 0; j < 4; j++) {
        lim[j] = plugin ? (q[j] > 0 ? cap : 0) : std::min<int64_t>(k, x[j]);
        total += lim[j];
        if (plugin)
            rho += q[j];
    }
    int h = std::min(k, plugin ? cap : total);
    std::vector<long double> w(h + 1, 0);
    if (plugin) {
        long double r = 1 - rho;
        if (r <= 0)
            return; // wrapper restricts plugin to rho<1
        w[0] = powl(r, k);
        for (int i = 0; i < h; i++)
            w[i + 1] = w[i] * (k - i) / r;
    } else {
        int64_t N = x[0] + x[1] + x[2] + x[3], r = L - N;
        int low = std::max<int64_t>(0, k - r);
        auto lc = [](int64_t n, int64_t z) {
            return lgammal(n + 1.L) - lgammal(z + 1.L) - lgammal(n - z + 1.L);
        };
        w[low] = expl(lc(r, k - low) - lc(L, k));
        for (int i = low; i < h; i++)
            w[i + 1] = w[i] * (k - i) / (r - k + i + 1.L);
    }
    std::vector<long double> c[4];
    for (int j = 0; j < 4; j++) {
        lim[j] = std::min(lim[j], h);
        c[j].resize(lim[j] + 1);
        c[j][0] = 1;
        for (int i = 0; i < lim[j]; i++)
            c[j][i + 1] = c[j][i] * (plugin ? q[j] : x[j] - i) / (i + 1.L);
    }
    int ns = std::min(h, lim[0] + lim[1] + lim[2]);
    int nt = lim[3];
    std::vector<long double> pm((ns + 1) * (nt + 1), 0), pt(pm.size(), 0);
    for (int s = 0; s <= ns; s++) {
        long double mass = 0, moment = 0;
        for (int t = 0; t <= nt; t++) {
            long double z = s + t <= h ? c[3][t] * w[s + t] : 0;
            mass += z;
            moment += t * z;
            pm[s * (nt + 1) + t] = mass;
            pt[s * (nt + 1) + t] = moment;
        }
    }
    long double f[3] = {0, 0, 0};
    for (int a = 0; a <= lim[0]; a++)
        for (int v = 0; v <= std::min(lim[1], h - a); v++)
            for (int b = 0; b <= std::min(lim[2], h - a - v); b++) {
                int s = a + v + b, off = s * (nt + 1);
                long double mass = pm[off + nt], pre = c[0][a] * c[1][v] * c[2][b];
                auto expected_min = [&](int bound, int shift) {
                    int cut = bound - shift;
                    if (cut <= 0)
                        return bound * mass;
                    int ix = std::min(cut, nt);
                    return shift * pm[off + ix] + pt[off + ix] + bound * (mass - pm[off + ix]);
                };
                f[0] += pre * expected_min(std::min(a + v, v + b), b);
                f[1] += pre * expected_min(v, b);
                f[2] += pre * (std::min(a, b) * mass + expected_min(v, 0));
            }
    for (int j = 0; j < 3; j++)
        ans[j] = (double)(f[j] / k);
}
extern "C" void fast5(const int64_t *X, const int64_t *L, const double *P, int n, const int *ks,
                      int nk, const int *caps, int mode, double *out) {
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++)
        for (int a = 0; a < nk; a++) {
            double q[4] = {P[i * 5], P[i * 5 + 1], P[i * 5 + 2], P[i * 5 + 3] + P[i * 5 + 4]};
            int64_t x[4] = {X[i * 5], X[i * 5 + 1], X[i * 5 + 2], X[i * 5 + 3] + X[i * 5 + 4]};
            double f[3], *o = out + (i * nk + a) * 5;
            e4(q, x, L[i], ks[a], caps[i * nk + a], mode, f);
            for (int j = 0; j < 3; j++)
                o[j] = f[j];
            q[3] = P[i * 5 + 4];
            x[3] = X[i * 5 + 4];
            e4(q, x, L[i], ks[a], caps[i * nk + a], mode, f);
            o[3] = f[0];
            q[3] = P[i * 5 + 3];
            x[3] = X[i * 5 + 3];
            e4(q, x, L[i], ks[a], caps[i * nk + a], mode, f);
            o[4] = f[0];
        }
}
