import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare


def p_matrix():
    p = np.zeros((5, 5), dtype=float)

    p[0] = [0.9915, 0.005, 0.0025, 0, 0.001]
    p[1] = [0, 0.986, 0.005, 0.004, 0.005]
    p[2] = [0, 0, 0.992, 0.003, 0.005]
    p[3] = [0, 0, 0, 0.991, 0.009]
    p[4] = [0, 0, 0, 0, 1.0]
    return p


P = p_matrix()
t = 120
n_women = 1000

p_0 = np.array([1, 0, 0, 0, 0])

# 1. Analytical Distribution (Expected Counts)
P_t = np.linalg.matrix_power(P, t)
analytical_dist = p_0 @ P_t
expected_counts = (
    analytical_dist * n_women
)  # Chi-square requires expected counts, not proportions

# 2. Simulated Distribution (Observed Counts)
simulated_states_at_t = np.zeros(5)

for _ in range(n_women):
    state = 0
    months = 0

    while months < t and state != 4:
        state = np.random.choice(5, p=P[state])
        months += 1

    simulated_states_at_t[state] += 1

# 3. Chi-Square Test
chi2_stat, p_value = chisquare(f_obs=simulated_states_at_t, f_exp=expected_counts)

# Print results
print(f"Validation at t = {t} months:\n")
print(f"Expected Counts: {np.round(expected_counts, 2)}")
print(f"Observed Counts: {simulated_states_at_t}\n")
print(f"Chi-square Statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value > 0.05:
    print(
        "Result: Fail to reject the null hypothesis. The simulation matches the analytical distribution."
    )
else:
    print("Result: Reject the null hypothesis. There is a significant difference.")
