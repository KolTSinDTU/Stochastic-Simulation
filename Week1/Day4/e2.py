import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SAMPLES_PER_ESTIMATE = 100
NUM_ESTIMATES = 10

def estimate_exponential():
    # 1. Generate uniform random variable
    u = rand.random()
    # 2. Return exponential transformation
    return math.exp(u)

def estimate_exponential_antithetic():
    # 1. Generate uniform random variable
    u = rand.random()
    # 2. Return average of antithetic pair
    return (math.exp(u) + math.exp(1 - u)) / 2

def calculate_confidence_interval(data, confidence_level=0.95):
    # 1. Initialize parameters
    num_samples = len(data)
    significance_alpha = 1 - confidence_level

    # 2. Calculate statistics
    sample_mean = np.mean(data)
    sample_std_deviation = np.std(data, ddof=1)

    # 3. Determine critical value and margin of error
    t_critical_value = stats.t.ppf(1 - significance_alpha / 2, df=num_samples - 1)
    margin_of_error = t_critical_value * (sample_std_deviation / math.sqrt(num_samples))

    # 4. Calculate and return interval bounds
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return float(lower_bound), float(upper_bound)

def get_mean_estimate_crude():
    # 1. Generate multiple crude estimates
    sample_estimates = [estimate_exponential() for _ in range(SAMPLES_PER_ESTIMATE)]
    # 2. Return mean
    return np.mean(sample_estimates)

def get_mean_estimate_antithetic():
    # 1. Generate multiple antithetic estimates
    sample_estimates = [estimate_exponential_antithetic() for _ in range(SAMPLES_PER_ESTIMATE)]
    # 2. Return mean
    return np.mean(sample_estimates)

def perform_crude_estimation():
    # 1. Generate collection of crude mean estimates
    mean_estimates_list = [get_mean_estimate_crude() for _ in range(NUM_ESTIMATES)]
    
    # 2. Calculate statistics
    overall_mean = np.mean(mean_estimates_list)
    overall_variance = np.var(mean_estimates_list, ddof=1)
    confidence_lower, confidence_upper = calculate_confidence_interval(mean_estimates_list)
    
    # 3. Output results
    print(f"--- Results: Crude Monte Carlo Estimation ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"Estimated Variance: {overall_variance:.4f}")
    print(f"95% Confidence Interval: [{confidence_lower:.4f}, {confidence_upper:.4f}]")

def perform_antithetic_estimation():
    # 1. Generate collection of antithetic mean estimates
    mean_estimates_list = [get_mean_estimate_antithetic() for _ in range(NUM_ESTIMATES)]
    
    # 2. Calculate statistics
    overall_mean = np.mean(mean_estimates_list)
    overall_variance = np.var(mean_estimates_list, ddof=1)
    confidence_lower, confidence_upper = calculate_confidence_interval(mean_estimates_list)
    
    # 3. Output results
    print(f"--- Results: Antithetic Variates Estimation ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"Estimated Variance: {overall_variance:.8f}")
    print(f"95% Confidence Interval: [{confidence_lower:.4f}, {confidence_upper:.4f}]")

if __name__ == "__main__":
    # 1. Run crude estimation
    perform_crude_estimation()
    print()
    # 2. Run antithetic estimation
    perform_antithetic_estimation()
