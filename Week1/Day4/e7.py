import random as rand
import numpy as np
import math

SAMPLES_COUNT = 1000


def get_standard_normal_sample():
    # 1. Return a sample from standard normal distribution
    return rand.gauss(0, 1)


def generate_exponential_sample(lambda_param):
    # 1. Generate random uniform
    u = rand.random()
    # 2. Return exponential sample
    return -math.log(u) / lambda_param


def calculate_crude_estimate(threshold):
    # 1. Initialize counter
    success_count = 0
    # 2. Perform trials
    for _ in range(SAMPLES_COUNT):
        if get_standard_normal_sample() > threshold:
            success_count += 1
    # 3. Return fraction
    return float(success_count / SAMPLES_COUNT)


def calculate_importance_sampling_estimate(threshold):
    # 1. Initialize estimates list
    trial_estimates = []
    # 2. Perform importance sampling
    for _ in range(SAMPLES_COUNT):
        # Sample from shifted distribution (mean = threshold)
        y_sample = rand.gauss(threshold, 1)
        if y_sample < threshold:
            trial_estimates.append(0)
        else:
            trial_estimates.append(math.exp((threshold**2) / 2 - threshold * y_sample))
    # 3. Return mean of estimates
    return np.mean(trial_estimates)


if __name__ == "__main__":
    threshold_value = 3

    # 1. Crude Monte Carlo
    print(f"--- Crude Monte Carlo Results ---")
    crude_results = [calculate_crude_estimate(threshold_value) for _ in range(10)]
    mean_crude = np.mean(crude_results)
    variance_crude = np.var(crude_results, ddof=1)
    print(f"Estimated Mean: {mean_crude:.6f}")
    print(f"Estimated Variance: {variance_crude:.8f}")
    print()

    # 2. Importance Sampling
    print(f"--- Importance Sampling Results ---")
    is_results = [calculate_importance_sampling_estimate(threshold_value) for _ in range(10)]
    mean_is = np.mean(is_results)
    variance_is = np.var(is_results, ddof=1)
    print(f"Estimated Mean: {mean_is:.6f}")
    print(f"Estimated Variance: {variance_is:.10f}")
    print()

    # 3. Summary
    print(f"--- Variance Reduction Summary ---")
    reduction_factor = variance_crude / variance_is if variance_is > 0 else 0
    print(f"Variance Reduction Factor: {reduction_factor:.4f}")
