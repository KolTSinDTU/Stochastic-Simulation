"""In this exercise you should implement everything including the tests
(e.g. the chi-square and KS tests) yourself. I recommend that you
also code routines for histogrammes yourself to better control
limits, but this is not strictly needed. Later, when your code is
working you are free to use builtin functions"""

"""Write a program implementing a linear congruential generator
(LCG). Be sure that the program works correctly using only
integer representation."""


import matplotlib.pyplot as plt
import random as rand

SAMPLES = 10_000

def lcg(seed, a, c, M, n):
    values = []
    x = seed

    for i in range(n):
        x = (a * x + c) % M
        values.append(x / M)

    return values

def chi_squared(samples, bins):
    T = 0
    n = len(samples)
    expected = n/bins
    
    observed = [0] * bins
    for u in samples:
        bin_index = int(u * bins)
        observed[bin_index] += 1

    for obs in observed:
        T += ((obs - expected) ** 2) / expected

    return T

def kolmogorov_smironv(samples):
    n = len(samples)
    sorted_samples = sorted(samples)

    d_plus = 0.0
    d_minus = 0.0

    for i in range(1, n + 1):
        x_i = sorted_samples[i - 1]

        # Difference just after x_i
        d_plus_i = i / n - x_i

        # Difference just before x_i
        d_minus_i = x_i - (i - 1) / n

        if d_plus_i > d_plus:
            d_plus = d_plus_i

        if d_minus_i > d_minus:
            d_minus = d_minus_i

    return max(d_plus, d_minus)

def use_builtin_rng(samples: int):
    return [rand.random() for _ in range(samples)]
            


bins = 100
seed = 3
a = 1664525
c = 1013904223
M = 2**32

random_numbers = lcg(seed, a, c, M, SAMPLES)
random_numbers_builtin = use_builtin_rng(SAMPLES)

T = chi_squared(random_numbers, bins)
T_builtin = chi_squared(random_numbers_builtin, bins)
print(T)
print(T_builtin)

ks_statistic = kolmogorov_smironv(random_numbers)
ks_statistic_builtin = kolmogorov_smironv(random_numbers_builtin)

print(ks_statistic)

# randints = [rand.randint for _  in range(SAMPLES)]
# plt.hist(random_numbers)
# plt.scatter(random_numbers, range(10000))
plt.scatter(random_numbers_builtin, range(10000))
# plt.scatter(range(10000), [rand.randint for _  in range(10000)])
plt.show()