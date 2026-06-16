import numpy as np
from scipy.stats import pareto

# Part 1: Ross, Chapter 8, Exercise 13
x_1 = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
n_1 = len(x_1)
x_bar_1 = np.mean(x_1)
B = 100000 # Number of bootstrap replicates

# Resample and calculate bootstrap sample means
boot_samples_1 = np.random.choice(x_1, size=(B, n_1), replace=True)
boot_means_1 = np.mean(boot_samples_1, axis=1)

# Estimate p = P{-5 < X_bar - \mu < 5} by replacing \mu with x_bar
# So we compute P{-5 < X_bar* - x_bar < 5}
p_estimate = np.mean((boot_means_1 - x_bar_1 > -5) & (boot_means_1 - x_bar_1 < 5))

# Part 2: Ross, Chapter 8, Exercise 15
x_2 = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])
n_2 = len(x_2)

# Resample and calculate bootstrap sample variances
boot_samples_2 = np.random.choice(x_2, size=(B, n_2), replace=True)
# ddof=1 for sample variance S^2
boot_variances_2 = np.var(boot_samples_2, axis=1, ddof=1)
var_S2_estimate = np.var(boot_variances_2, ddof=1)

# Part 3: Custom Exercise (Pareto)
np.random.seed(42) # For reproducibility in explanation
n_3 = 200
shape_k = 1.05
scale_beta = 1.0

# Generate Pareto sample
# scipy.stats.pareto uses 'b' for shape.
sample_3 = pareto.rvs(b=shape_k, scale=scale_beta, size=n_3)

sample_mean_3 = np.mean(sample_3)
sample_median_3 = np.median(sample_3)

# Bootstrap for Part 3 with k=100
k_boot = 100
boot_samples_3 = np.random.choice(sample_3, size=(k_boot, n_3), replace=True)
boot_means_3 = np.mean(boot_samples_3, axis=1)
boot_medians_3 = np.median(boot_samples_3, axis=1)

boot_var_mean = np.var(boot_means_3, ddof=1)
boot_var_median = np.var(boot_medians_3, ddof=1)

print(f"--- Part 1 ---")
print(f"Sample mean (x_bar): {x_bar_1}")
print(f"Estimated p: {p_estimate}")

print(f"\n--- Part 2 ---")
print(f"Estimated Var(S^2): {var_S2_estimate}")

print(f"\n--- Part 3 ---")
print(f"Sample mean: {sample_mean_3}")
print(f"Sample median: {sample_median_3}")
print(f"Bootstrap Var(Mean): {boot_var_mean}")
print(f"Bootstrap Var(Median): {boot_var_median}")