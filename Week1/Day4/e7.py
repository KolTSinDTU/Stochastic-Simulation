import random as rand
import statistics as stats
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

SAMPLES = 1000


def standard_normal():
    return rand.gauss(0, 1)


def exponential_distribution(lam):
    u = rand.random()
    return -math.log(u) / lam


def crude_estimate(a):
    count = 0
    for _ in range(SAMPLES):
        if standard_normal() > a:
            count += 1
    return float(count / SAMPLES)


def importance_sampling_estimate(a):
    estimates = []
    for _ in range(SAMPLES):
        y = rand.gauss(a, 1)
        if y < a:
            estimates.append(0)
        else:
            estimates.append(math.exp((a**2) / 2 - a * y))
    mean_estimate = np.mean(estimates)
    return mean_estimate


if __name__ == "__main__":
    a = 3

    print("Crude Monte Carlo Estimation:")
    crude_estimates = [crude_estimate(a) for _ in range(10)]
    mean_crude = np.mean(crude_estimates)
    vars_crude = np.var(crude_estimates, ddof=1)
    print(f"Estimated Mean Crude: {mean_crude}")
    print(f"Estimated Variance Crude: {vars_crude}")

    print("\nImportance Sampling Estimation:")
    is_estimates = [importance_sampling_estimate(a) for _ in range(10)]
    mean_is = np.mean(is_estimates)
    vars_is = np.var(is_estimates, ddof=1)
    print(f"Estimated Mean IS: {mean_is}")
    print(f"Estimated Variance IS: {vars_is:.10f}")

    print("\nVariance Reduction:")
    print(f"Variance Reduction: {vars_crude / vars_is:.10f}")
