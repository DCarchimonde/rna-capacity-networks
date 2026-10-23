// Exact with-replacement plug-in at fixed k. No Monte Carlo integration.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
struct Entry {
    std::array<int, 6> z;
    double coefficient;
    std::array<double, 5> f;
};
static void build(int category, int left, std::array<int, 6> &z, int k, bool common,
                  const std::vector<double> &fact, std::vector<Entry> &out) {
    if (category < 5) {
        if (common && category == 3) {
            z[3] = 0;
            build(category + 1, left, z, k, common, fact, out);
            return;
        }
        for (int a = 0; a <= left; a++) {
            z[category] = a;
            build(category + 1, left - a, z, k, common, fact, out);
        }
        return;
    }
    z[5] = left;
    Entry e;
    e.z = z;
    e.coefficient = fact[k];
    for (int a : z)
        e.coefficient /= fact[a];
    double a = z[0], v = z[1], b = z[2], c = z[3], d = z[4];
    e.f[0] = std::min({a + v, v + b, b + c + d}) / k;
    e.f[1] = std::min(v, b + c + d) / k;
    e.f[2] = (std::min(a, b) + std::min(v, c + d)) / k;
    if (common) {
        e.f[3] = std::min({a + v, v + b, b}) / k;
        e.f[4] = 0;
    } else {
        e.f[3] = std::min({a + v, v + b, b + d}) / k;
        e.f[4] = std::min({a + v, v + b, b + c}) / k;
    }
    out.push_back(e);
}
extern "C" void multinomial_plugin(const double *probabilities, int n, int k, int common,
                                   double *output, double *mass_error) {
    std::vector<double> fact(k + 1, 1);
    for (int i = 1; i <= k; i++)
        fact[i] = fact[i - 1] * i;
    std::vector<Entry> entries;
    std::array<int, 6> z{};
    build(0, k, z, k, common, fact, entries);
    const int nf = common ? 4 : 5;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double powers[6][33];
        double psum = 0;
        for (int j = 0; j < 5; j++) {
            double p = (common && j == 3) ? 0 : probabilities[i * 5 + j];
            psum += p;
            powers[j][0] = 1;
            for (int a = 1; a <= k; a++)
                powers[j][a] = powers[j][a - 1] * p;
        }
        powers[5][0] = 1;
        for (int a = 1; a <= k; a++)
            powers[5][a] = powers[5][a - 1] * (1 - psum);
        for (int f = 0; f < nf; f++)
            output[i * nf + f] = 0;
        double mass = 0;
        for (const auto &e : entries) {
            double w = e.coefficient;
            for (int j = 0; j < 6; j++)
                w *= powers[j][e.z[j]];
            mass += w;
            for (int f = 0; f < nf; f++)
                output[i * nf + f] += w * e.f[f];
        }
        mass_error[i] = std::abs(mass - 1);
    }
}
