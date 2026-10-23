#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

static double logchoose(int64_t n, int64_t r) {
    if (r < 0 || r > n)
        return -INFINITY;
    r = std::min(r, n - r);
    if (r == 0)
        return 0.;
    // Small r avoids cancellation in lgamma for large whole-cell depths.
    if (r <= 32) {
        double out = 0.;
        for (int64_t j = 1; j <= r; ++j)
            out += std::log(double(n - r + j)) - std::log(double(j));
        return out;
    }
    return std::lgamma(double(n) + 1.) - std::lgamma(double(r) + 1.) -
           std::lgamma(double(n - r) + 1.);
}

static double pair_mean(int64_t x, int64_t y, int64_t L, int k) {
    if (x == 0 || y == 0 || k < 2)
        return 0.;
    if (k == L)
        return double(std::min(x, y)) / k;
    if (k == 2)
        return double(x) * double(y) / (double(L) * double(L - 1));
    if (x > y)
        std::swap(x, y);
    int64_t rest = L - x - y;
    const double den = logchoose(L, k);
    double answer = 0.;
    for (int64_t a = 1; a <= std::min(x, int64_t(k - 1)); ++a) {
        int64_t blo = std::max(int64_t(1), int64_t(k) - rest - a);
        int64_t bhi = std::min(y, int64_t(k) - a);
        if (blo > bhi)
            continue;
        double logp = logchoose(x, a) + logchoose(y, blo) + logchoose(rest, k - a - blo) - den;
        double p = std::exp(logp);
        for (int64_t b = blo; b <= bhi; ++b) {
            answer += double(std::min(a, b)) * p;
            if (b < bhi) {
                double ratio = (double(y - b) / double(b + 1)) *
                               (double(k - a - b) / double(rest - k + a + b + 1));
                p *= ratio;
                // Restart when the initial tail underflowed; otherwise a
                // recurrence could stay zero even after returning to support.
                if (p == 0.)
                    p = std::exp(logchoose(x, a) + logchoose(y, b + 1) +
                                 logchoose(rest, k - a - b - 1) - den);
            }
        }
    }
    return answer / k;
}

extern "C" void pair_flows(const int64_t *X, const int64_t *L, int n, int d, const int32_t *edges,
                           int ne, const int32_t *ks, int nk, double *out) {
#pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n; ++i) {
        for (int h = 0; h < nk; ++h)
            for (int e = 0; e < ne; ++e) {
                out[(int64_t(i) * nk + h) * ne + e] =
                    pair_mean(X[int64_t(i) * d + edges[2 * e]],
                              X[int64_t(i) * d + edges[2 * e + 1]], L[i], ks[h]);
            }
    }
}

static double multinomial_pair(double p, double q, int k) {
    if (p <= 0. || q <= 0.)
        return 0.;
    double t = p + q, r = p / t;
    int upper = k;
    if (t < 1.) {
        double lambda = (k - 1) * t;
        for (int m = std::max(2, int(std::ceil(lambda)) + 1); m < k; ++m) {
            double bound = .5 * t * std::exp(-m * std::log(m / lambda) + m - lambda);
            if (bound <= 1e-16) {
                upper = m;
                break;
            }
        }
    }
    double out = 0.;
    for (int n = 2; n <= upper; ++n) {
        if (t == 1. && n != k)
            continue;
        double pn =
            t == 1. ? 1. : std::exp(logchoose(k, n) + n * std::log(t) + (k - n) * std::log1p(-t));
        double conditional = 0.;
        for (int a = 1; a < n; ++a) {
            double pa = std::exp(logchoose(n, a) + a * std::log(r) + (n - a) * std::log1p(-r));
            conditional += std::min(a, n - a) * pa;
        }
        out += pn * conditional / k;
    }
    return out;
}

extern "C" void multinomial_pairs(const double *P, int n, int k, double *out) {
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; ++i)
        out[i] = multinomial_pair(P[2 * i], P[2 * i + 1], k);
}
