import numpy as np


def p_matrix():
    p = np.zeros((5, 5), dtype=float)

    p[0] = [0.9915, 0.005, 0.0025, 0, 0.001]
    p[1] = [0, 0.986, 0.005, 0.004, 0.005]
    p[2] = [0, 0, 0.992, 0.003, 0.005]
    p[3] = [0, 0, 0, 0.991, 0.009]
    p[4] = [0, 0, 0, 0, 1.0]
    return p


P = p_matrix()

# Calculate Exact Theoretical Expected Lifetime (E[X])
pi = np.array([1, 0, 0, 0])
P_s = P[0:4, 0:4]
I = np.eye(4)
expected_mean = pi @ np.linalg.inv(I - P_s) @ np.ones(4)

# Run crude Monte Carlo Simulation
n_batches = 100
n_women = 200
threshold = 350

Y_crude = np.zeros(n_batches)  # Y: Fraction dying <= 350 months
X_means = np.zeros(n_batches)  # X: Mean lifetime of the batch

for i in range(n_batches):
    lifetimes = np.zeros(n_women)

    for j in range(n_women):
        state = 0
        months = 0
        while state != 4:
            state = np.random.choice(5, p=P[state])
            months += 1
        lifetimes[j] = months

    # Record the fraction and the mean for this batch
    Y_crude[i] = np.mean(lifetimes <= threshold)
    X_means[i] = np.mean(lifetimes)

# Apply Control Variates
# Covariance between X and Y to find optimal c
cov_matrix = np.cov(X_means, Y_crude)
c_star = cov_matrix[0, 1] / cov_matrix[0, 0]  # Cov(X,Y) / Var(X)

# Calculate the adjusted estimator Y_cv
Y_cv = Y_crude - c_star * (X_means - expected_mean)

# Compare Variances
var_crude = np.var(Y_crude, ddof=1)
var_cv = np.var(Y_cv, ddof=1)

variance_reduction = (1 - (var_cv / var_crude)) * 100

print(f"Theoretical Mean Lifetime (E[X]): {expected_mean:.2f} months\n")
print(f"Crude Estimator Mean: {np.mean(Y_crude):.4f}")
print(f"Crude Estimator Variance: {var_crude:.6f}\n")

print(f"CV Estimator Mean: {np.mean(Y_cv):.4f}")
print(f"CV Estimator Variance: {var_cv:.6f}\n")

print(f"Variance Reduction: {variance_reduction:.2f}%")
