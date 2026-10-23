#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Counts of the four effective vertex categories a, v, b, and c+d.
// Remaining molecules are a fifth category. We enumerate the exact
// multivariate hypergeometric law; no Monte Carlo or support truncation.
static void expect4(int64_t a, int64_t v, int64_t b, int64_t t, int64_t L, int k, double *result,
                    double *mass_error) {
    result[0] = result[1] = result[2] = 0;
    if (a + v == 0 || b + t == 0) {
        *mass_error = 0;
        return;
    }
    const int64_t N = a + v + b + t, rest = L - N;
    const int low = std::max<int64_t>(0, k - rest);
    const int high = std::min<int64_t>(k, N);
    std::vector<long double> wr(k + 1, 0);
    auto lc = [](int64_t n, int64_t r) {
        return lgammal(n + 1.L) - lgammal(r + 1.L) - lgammal(n - r + 1.L);
    };
    wr[low] = expl(lc(rest, k - low) - lc(L, k));
    for (int s = low; s < high; s++)
        wr[s + 1] = wr[s] * (k - s) / (rest - k + s + 1.L);
    std::vector<std::vector<long double>> cs(4);
    int64_t counts[4] = {a, v, b, t};
    for (int j = 0; j < 4; j++) {
        int m = std::min<int64_t>(counts[j], k);
        cs[j].resize(m + 1);
        cs[j][0] = 1;
        for (int z = 0; z < m; z++)
            cs[j][z + 1] = cs[j][z] * (counts[j] - z) / (z + 1.L);
    }
    long double total = 0, f0 = 0, fab = 0, fvb = 0;
    for (int aa = 0; aa < (int)cs[0].size(); aa++)
        for (int vv = 0; vv < std::min<int>(cs[1].size(), k - aa + 1); vv++)
            for (int bb = 0; bb < std::min<int>(cs[2].size(), k - aa - vv + 1); bb++) {
                const long double pre = cs[0][aa] * cs[1][vv] * cs[2][bb];
                for (int tt = 0; tt < std::min<int>(cs[3].size(), k - aa - vv - bb + 1); tt++) {
                    const long double w = pre * cs[3][tt] * wr[aa + vv + bb + tt];
                    total += w;
                    f0 += w * std::min({aa + vv, vv + bb, bb + tt});
                    fab += w * std::min(vv, bb + tt);
                    fvb += w * (std::min(aa, bb) + std::min(vv, tt));
                }
            }
    result[0] = (double)(f0 / k);
    result[1] = (double)(fab / k);
    result[2] = (double)(fvb / k);
    *mass_error = (double)fabsl(total - 1.L);
}

extern "C" void flow_resolution(const int64_t *x, const int64_t *depth, int n, const int *ks,
                                int nk, int common, double *out, double *errors) {
    const int nf = common ? 4 : 5;
#pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < n; i++) {
        const int64_t *z = x + 5 * i;
        for (int q = 0; q < nk; q++) {
            const int k = ks[q];
            double *row = out + (i * nk + q) * nf;
            double r[3], err = 0, err2 = 0;
            expect4(z[0], z[1], z[2], common ? z[4] : z[3] + z[4], depth[i], k, r, &err);
            for (int j = 0; j < 3; j++)
                row[j] = r[j];
            if (common) {
                expect4(z[0], z[1], z[2], 0, depth[i], k, r, &err2);
                row[3] = r[0];
                err = std::max(err, err2);
            } else {
                expect4(z[0], z[1], z[2], z[4], depth[i], k, r, &err2);
                row[3] = r[0];
                err = std::max(err, err2);
                expect4(z[0], z[1], z[2], z[3], depth[i], k, r, &err2);
                row[4] = r[0];
                err = std::max(err, err2);
            }
            errors[i * nk + q] = err;
        }
    }
}
