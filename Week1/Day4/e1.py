import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SAMPLES_PER_ESTIMATE = 100
NUM_ESTIMATES = 20

def estimate_exponential_function():
    # 1. Generate uniform random variable
    u = rand.random()
    # 2. Return exponential transformation
    return math.exp(u)

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

def get_mean_estimate():
    # 1. Generate multiple estimates
    sample_estimates = [estimate_exponential_function() for _ in range(SAMPLES_PER_ESTIMATE)]
    # 2. Calculate and return mean
    return np.mean(sample_estimates)

if __name__ == "__main__":
    # 1. Generate collection of mean estimates
    mean_estimates_list = [get_mean_estimate() for _ in range(NUM_ESTIMATES)]
    
    # 2. Calculate overall statistics
    overall_mean = np.mean(mean_estimates_list)
    confidence_lower, confidence_upper = calculate_confidence_interval(mean_estimates_list)
    
    # 3. Output results
    print(f"--- Results: Monte Carlo Estimation ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"95% Confidence Interval: [{confidence_lower:.4f}, {confidence_upper:.4f}]")
