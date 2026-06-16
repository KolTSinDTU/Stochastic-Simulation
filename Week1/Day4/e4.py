import random as rand
import math
import numpy as np
from scipy import stats

SAMPLES_COUNT = 100


def estimate_exponential_value():
    # 1. Generate a random uniform number
    u = rand.random()
    # 2. Calculate exponential value
    return math.exp(u)


def estimate_exponential_antithetic():
    # 1. Generate a random uniform number
    u = rand.random()
    # 2. Return average of antithetic pair
    return (math.exp(u) + math.exp(1 - u)) / 2


def estimate_exponential_control_variate():
    # 1. Generate a random uniform number
    u = rand.random()
    # 2. Return estimate with control variate adjustment
    return math.exp(u) - 1.69 * (u - 1 / 2)


def get_mean_estimate_stratified():
    # 1. Initialize strata parameters
    num_strata = 10
    samples_per_stratum = SAMPLES_COUNT // num_strata
    strata_estimates = []

    # 2. Sample from each stratum
    for i in range(num_strata):
        stratum_samples = []
        for _ in range(samples_per_stratum):
            u = rand.random() * (1 / num_strata) + (i / num_strata)
            stratum_samples.append(math.exp(u))
        strata_estimates.append(np.mean(stratum_samples))

    # 3. Return overall mean
    return np.mean(strata_estimates)


def calculate_confidence_interval(data_points, confidence_level=0.95):
    # 1. Calculate basic statistics
    n_samples = len(data_points)
    alpha_value = 1 - confidence_level
    sample_mean = np.mean(data_points)
    sample_std_dev = np.std(data_points, ddof=1)

    # 2. Determine critical t-value
    t_critical_value = stats.t.ppf(1 - alpha_value / 2, df=n_samples - 1)

    # 3. Calculate margin of error
    margin_of_error_val = t_critical_value * (sample_std_dev / math.sqrt(n_samples))

    # 4. Define bounds
    lower_bound = sample_mean - margin_of_error_val
    upper_bound = sample_mean + margin_of_error_val
    return float(lower_bound), float(upper_bound)


def get_crude_mean_estimate():
    # 1. Collect samples
    simulation_results = [estimate_exponential_value() for _ in range(SAMPLES_COUNT)]
    # 2. Calculate mean
    return np.mean(simulation_results)


def get_antithetic_mean_estimate():
    # 1. Collect samples
    simulation_results = [estimate_exponential_antithetic() for _ in range(SAMPLES_COUNT)]
    # 2. Calculate mean
    return np.mean(simulation_results)


def get_control_variate_mean_estimate():
    # 1. Collect samples
    simulation_results = [estimate_exponential_control_variate() for _ in range(SAMPLES_COUNT)]
    # 2. Calculate mean
    return np.mean(simulation_results)


def run_crude_monte_carlo():
    # 1. Perform multiple trials
    trial_estimates = [get_crude_mean_estimate() for _ in range(10)]
    # 2. Calculate summary statistics
    overall_mean = np.mean(trial_estimates)
    ci_lower, ci_upper = calculate_confidence_interval(trial_estimates)
    
    # 3. Output results
    print(f"--- Crude Monte Carlo Results ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"Estimated variance: {np.var(trial_estimates, ddof=1):.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


def run_antithetic_monte_carlo():
    # 1. Perform multiple trials
    trial_estimates = [get_antithetic_mean_estimate() for _ in range(10)]
    # 2. Calculate summary statistics
    overall_mean = np.mean(trial_estimates)
    ci_lower, ci_upper = calculate_confidence_interval(trial_estimates)
    
    # 3. Output results
    print(f"--- Antithetic Variates Results ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"Estimated variance: {np.var(trial_estimates, ddof=1):.8f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


def run_control_variate_monte_carlo():
    # 1. Perform multiple trials
    trial_estimates = [get_control_variate_mean_estimate() for _ in range(10)]
    # 2. Calculate summary statistics
    overall_mean = np.mean(trial_estimates)
    ci_lower, ci_upper = calculate_confidence_interval(trial_estimates)
    
    # 3. Output results
    print(f"--- Control Variate Results ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"Estimated variance: {np.var(trial_estimates, ddof=1):.8f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


def run_stratified_monte_carlo():
    # 1. Perform multiple trials
    trial_estimates = [get_mean_estimate_stratified() for _ in range(10)]
    # 2. Calculate summary statistics
    overall_mean = np.mean(trial_estimates)
    ci_lower, ci_upper = calculate_confidence_interval(trial_estimates)
    
    # 3. Output results
    print(f"--- Stratified Sampling Results ---")
    print(f"Estimated Mean: {overall_mean:.4f}")
    print(f"Estimated variance: {np.var(trial_estimates, ddof=1):.8f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


if __name__ == "__main__":
    run_crude_monte_carlo()
    print()
    run_antithetic_monte_carlo()
    print()
    run_control_variate_monte_carlo()
    print()
    run_stratified_monte_carlo()
