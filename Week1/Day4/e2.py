import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SAMPLES = 100

def estimate_exp():
    u = rand.random()
    return math.exp(u)

def estimate_exp_antithetic():
    u = rand.random()
    return (math.exp(u) + math.exp(1 - u)) / 2

def confidence_interval(data, confidence=0.95):
    n = len(data)
    alpha = 1 - confidence

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # Calculate margin of error using the critical value, not the confidence percentage
    margin_of_error = t_critical * (std_dev / math.sqrt(n))

    lower_ci = mean - margin_of_error
    upper_ci = mean + margin_of_error
    return float(lower_ci), float(upper_ci)

def get_mean_estimate():
    estimates = [estimate_exp() for _ in range(SAMPLES)]
    mean_estimate = np.mean(estimates)
    return mean_estimate

def get_mean_estimate_antithetic():
    estimates = [estimate_exp_antithetic() for _ in range(SAMPLES)]
    mean_estimate = np.mean(estimates)
    return mean_estimate

def get_confidence_interval_crude():
    estimates = [get_mean_estimate() for _ in range(10)]
    mean_of_estimates = np.mean(estimates)
    ci_lower, ci_upper = confidence_interval(estimates)
    print(f"Estimated Mean: {mean_of_estimates:.4f}")
    print(f"Estimated variance: {np.var(estimates, ddof=1):.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

def get_confidence_interval_antithetic():
    estimates = [get_mean_estimate_antithetic() for _ in range(10)]
    mean_of_estimates = np.mean(estimates)
    ci_lower, ci_upper = confidence_interval(estimates)
    print(f"Estimated Mean (Antithetic): {mean_of_estimates:.4f}")
    print(f"Estimated variance (Antithetic): {np.var(estimates, ddof=1):.8f}")
    print(f"95% Confidence Interval (Antithetic): [{ci_lower:.4f}, {ci_upper:.4f}]")

if __name__ == "__main__":
    print("Crude Monte Carlo Estimation:")
    get_confidence_interval_crude()
    
    print("\nAntithetic Variates Estimation:")
    get_confidence_interval_antithetic()