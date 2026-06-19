import numpy as np
import matplotlib.pyplot as plt


def p_matrix():
    p = np.zeros((5, 5), dtype=float)

    p[0] = [0.9915, 0.005, 0.0025, 0, 0.001]
    p[1] = [0, 0.986, 0.005, 0.004, 0.005]
    p[2] = [0, 0, 0.992, 0.003, 0.005]
    p[3] = [0, 0, 0, 0.991, 0.009]
    p[4] = [0, 0, 0, 0, 1.0]
    return p


P = p_matrix()

pi = np.array([1, 0, 0, 0])  # Initial distribution over transient states (States 1-4)
P_s = P[0:4, 0:4] 
p_s = P[0:4, 4]  # Column vector of death probabilities
I = np.eye(4)
ones = np.ones(4)

# Calculate Theoretical Mean Lifetime
inv_matrix = np.linalg.inv(I - P_s)
expected_mean = pi @ inv_matrix @ ones

# Run Simulation
n_women = 1000
lifetimes = []

for _ in range(n_women):
    state = 0
    months = 0
    while state != 4:
        state = np.random.choice(5, p=P[state])
        months += 1
    lifetimes.append(months)

simulated_mean = np.mean(lifetimes)

# Print Mean Comparison
print(f"Theoretical Mean Lifetime: {expected_mean:.2f} months")
print(f"Simulated Mean Lifetime:   {simulated_mean:.2f} months\n")

# Calculate Theoretical PMF Curve
max_t = max(lifetimes)
t_values = np.arange(1, max_t + 1)
theoretical_pmf = np.zeros(max_t)

for t in t_values:
    P_s_t_minus_1 = np.linalg.matrix_power(P_s, t - 1)
    theoretical_pmf[t - 1] = pi @ P_s_t_minus_1 @ p_s

plt.figure(figsize=(10, 6))

# Use density=True to convert the histogram counts to probabilities
plt.hist(lifetimes, bins=70, density=True, edgecolor='black', alpha=0.6, label='Simulated Probability')

# Plot the theoretical PMF directly
plt.plot(t_values, theoretical_pmf, color='red', linewidth=2, label='Theoretical PMF')

plt.title("Lifetime Distribution: Simulated vs Theoretical Probabilities")
plt.xlabel("Months until Death (t)")
plt.ylabel("Probability P(T = t)")
plt.xlim(0, 1000) 
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()