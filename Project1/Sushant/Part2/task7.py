import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generator matrix (5x5)
Q = np.array(
    [
        [-0.0085, 0.005, 0.0025, 0, 0.001],
        [0, -0.014, 0.005, 0.004, 0.005],
        [0, 0, -0.008, 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0],
    ]
)

n_women = 1000
lifetimes = np.zeros(n_women)
distant_by_30_5 = 0

for i in range(n_women):
    state = 0  # Start in State 1 (index 0)
    current_time = 0.0
    had_distant = False

    while state != 4:  # Loop until Death (index 4)
        lambda_i = -Q[state, state]

        # Sample time spent in the current state
        dt = np.random.exponential(1 / lambda_i)
        current_time += dt

        # Form transition probabilities
        probs = Q[state, :].copy()
        probs[state] = 0
        probs = probs / lambda_i

        # Determine next state
        next_state = np.random.choice(5, p=probs)

        if next_state in [2, 3] and current_time > 30.5 and not had_distant:
            had_distant = True

        state = next_state

    lifetimes[i] = current_time
    if had_distant:
        distant_by_30_5 += 1

mean_lifetime = np.mean(lifetimes)
std_lifetime = np.std(lifetimes, ddof=1)

# 95% Confidence Interval for Mean (Normal Approximation)
alpha = 0.05
z_score = stats.norm.ppf(1 - alpha / 2)
margin_of_error = z_score * (std_lifetime / np.sqrt(n_women))
mean_ci = (mean_lifetime - margin_of_error, mean_lifetime + margin_of_error)

# 95% Confidence Interval for Standard Deviation (Chi-Square)
df = n_women - 1
chi2_lower = stats.chi2.ppf(alpha / 2, df)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)

var_lifetime = std_lifetime**2
var_ci_lower = (df * var_lifetime) / chi2_upper
var_ci_upper = (df * var_lifetime) / chi2_lower
std_ci = (np.sqrt(var_ci_lower), np.sqrt(var_ci_upper))

proportion_distant = distant_by_30_5 / n_women

print(f"Mean Lifetime:       {mean_lifetime:.2f} months")
print(f"95% CI for Mean:     [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]\n")

print(f"Standard Deviation:  {std_lifetime:.2f} months")
print(f"95% CI for Std Dev:  [{std_ci[0]:.2f}, {std_ci[1]:.2f}]\n")

print(f"Proportion with distant recurrence after 30.5 months: {proportion_distant:.4f}")

plt.figure(figsize=(10, 6))
plt.hist(lifetimes, bins=40, edgecolor="black", alpha=0.7)
plt.title("Lifetime Distribution After Surgery (CTMC Simulation)")
plt.xlabel("Months until Death")
plt.ylabel("Number of Women")
plt.grid(axis="y", alpha=0.5)
plt.show()
