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
    # 1. Initialize variables
    values = []
    x = seed

    # 2. Generate random numbers using LCG formula
    for i in range(n):
        x = (a * x + c) % M
        values.append(x / M)

    return values

def chi_squared(samples, num_bins):
    # 1. Initialize calculation parameters
    total_chi_sq = 0
    n = len(samples)
    expected_count = n / num_bins
    
    # 2. Count observations in each bin
    observed_counts = [0] * num_bins
    for u in samples:
        bin_index = int(u * num_bins)
        if bin_index == num_bins: # Handle the case where u = 1.0
            bin_index -= 1
        observed_counts[bin_index] += 1

    # 3. Calculate Chi-Squared statistic
    for observed in observed_counts:
        total_chi_sq += ((observed - expected_count) ** 2) / expected_count

    return total_chi_sq

def kolmogorov_smirnov(samples):
    # 1. Sort samples and initialize statistics
    n = len(samples)
    sorted_samples = sorted(samples)

    d_plus = 0.0
    d_minus = 0.0

    # 2. Iterate through samples to find maximum deviations
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

def use_builtin_rng(num_samples: int):
    # 1. Generate samples using built-in random module
    return [rand.random() for _ in range(num_samples)]
            

if __name__ == "__main__":
    # 1. Define LCG parameters
    num_bins = 100
    seed_value = 3
    multiplier = 1664525
    increment = 1013904223
    modulus = 2**32

    # 2. Generate samples
    random_numbers = lcg(seed_value, multiplier, increment, modulus, SAMPLES)
    random_numbers_builtin = use_builtin_rng(SAMPLES)

    # 3. Perform statistical tests
    chi_sq_lcg = chi_squared(random_numbers, num_bins)
    chi_sq_builtin = chi_squared(random_numbers_builtin, num_bins)
    
    ks_stat_lcg = kolmogorov_smirnov(random_numbers)
    ks_stat_builtin = kolmogorov_smirnov(random_numbers_builtin)

    # 4. Print results
    print(f"--- Results: Chi-Squared Test ---")
    print(f"LCG Chi-Squared: {chi_sq_lcg:.4f}")
    print(f"Built-in Chi-Squared: {chi_sq_builtin:.4f}")
    print()
    print(f"--- Results: Kolmogorov-Smirnov Test ---")
    print(f"LCG KS Statistic: {ks_stat_lcg:.4f}")
    print(f"Built-in KS Statistic: {ks_stat_builtin:.4f}")

    # 5. Visualize results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(random_numbers_builtin, range(SAMPLES), color='#2c3e50', alpha=0.5, s=1, label='Built-in RNG')
    ax.set_title("Built-in Random Number Generator Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Sample Index")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
