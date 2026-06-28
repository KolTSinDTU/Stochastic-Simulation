import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Setup
np.random.seed(42)
RHO = 0.5
COV_MATRIX = np.array([[1, RHO], [RHO, 1]])

# --- Part (a): Generate prior sample ---
# Generate (xi, gamma) from Bivariate Normal(0, Cov)
xi_prior, gamma_prior = np.random.multivariate_normal([0, 0], COV_MATRIX)

# Transform to (theta, psi)
theta_true = np.exp(xi_prior)
psi_true = np.exp(gamma_prior)

print(f"Part (a) True values sampled from prior:")
print(f"Theta: {theta_true:.4f}, Psi (Variance): {psi_true:.4f}\n")


# --- Part (b): Simulate observations ---
def simulate_data(n, theta, psi):
    # N(Theta, Psi) where Psi is variance. Scipy/Numpy use std dev (sqrt(Psi))
    return np.random.normal(loc=theta, scale=np.sqrt(psi), size=n)


# --- Part (d) & (e): Metropolis Hastings ---
def log_posterior_logspace(xi, gamma, n, x_bar, s2):
    """Calculates unnormalized log-posterior in the unconstrained (Xi, Gamma) space."""
    theta = np.exp(xi)
    psi = np.exp(gamma)

    # 1. Log-prior of (Xi, Gamma) ~ Bivariate Normal
    log_prior = -0.5 * (xi**2 - 2 * RHO * xi * gamma + gamma**2) / (1 - RHO**2)

    # 2. Log-likelihood utilizing the Remark for O(1) computation
    sum_sq = (n - 1) * s2 + n * (x_bar - theta) ** 2
    log_likelihood = -(n / 2) * np.log(psi) - (sum_sq / (2 * psi))

    return log_prior + log_likelihood


def run_mh(data, iterations=10000, burn_in=2000, step_size=0.4):
    n = len(data)
    x_bar = np.mean(data)
    s2 = np.var(data, ddof=1) if n > 1 else 0

    # Initialize chain near the sample estimates
    xi_curr = np.log(x_bar) if x_bar > 0 else 0
    gamma_curr = np.log(s2) if s2 > 0 else 0

    log_p_curr = log_posterior_logspace(xi_curr, gamma_curr, n, x_bar, s2)

    samples = np.zeros((iterations, 2))
    accepted = 0

    for i in range(iterations):
        # Symmetric Random Walk Proposal in unconstrained space
        xi_prop = xi_curr + np.random.normal(0, step_size)
        gamma_prop = gamma_curr + np.random.normal(0, step_size)

        log_p_prop = log_posterior_logspace(xi_prop, gamma_prop, n, x_bar, s2)

        # Metropolis acceptance criterion (symmetric proposal -> Hastings ratio = 1)
        alpha = np.exp(log_p_prop - log_p_curr)

        if np.random.rand() < alpha:
            xi_curr, gamma_curr = xi_prop, gamma_prop
            log_p_curr = log_p_prop
            accepted += 1

        samples[i] = [xi_curr, gamma_curr]

    print(f"Acceptance Rate (n={n}): {accepted/iterations:.2f}")

    # Discard burn-in and transform back to Theta, Psi
    theta_samples = np.exp(samples[burn_in:, 0])
    psi_samples = np.exp(samples[burn_in:, 1])

    return theta_samples, psi_samples


# --- Explicitly generating and printing Part (b) ---
data_part_b = simulate_data(10, theta_true, psi_true)
print("Part (b) Simulated Data (n=10):")
print(np.array2string(data_part_b, precision=4, separator=", "))
print("\n")

# --- Part (d): Run Metropolis-Hastings for n=10 ---
print("Executing Part (d): Running MH algorithm on n=10 data...")

# We use the run_mh() function defined earlier, which contains the MCMC loop,
# the log-space transformations, and the acceptance/rejection logic.
theta_samples_d, psi_samples_d = run_mh(data_part_b, iterations=10000, burn_in=2000)

print(f"Part (d) Posterior Mean Theta: {np.mean(theta_samples_d):.4f}")
print(f"Part (d) Posterior Mean Psi: {np.mean(psi_samples_d):.4f}\n")

# Optional but highly recommended: Plot the posterior distributions for Part (d)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(theta_samples_d, bins=50, color="skyblue", edgecolor="black", density=True)
ax1.axvline(
    theta_true,
    color="red",
    linestyle="dashed",
    linewidth=2,
    label=f"True Theta: {theta_true:.2f}",
)
ax1.set_title("Posterior Distribution of Theta (n=10)")
ax1.legend()

ax2.hist(psi_samples_d, bins=50, color="salmon", edgecolor="black", density=True)
ax2.axvline(
    psi_true,
    color="red",
    linestyle="dashed",
    linewidth=2,
    label=f"True Psi: {psi_true:.2f}",
)
ax2.set_title("Posterior Distribution of Psi (n=10)")
ax2.legend()

plt.tight_layout()
plt.show()

# Execute Part (e) parameters
n_values = [10, 100, 1000]
results = {}


for n in n_values:
    # Use the same true parameters from Part (a) to generate new datasets
    data = simulate_data(n, theta_true, psi_true)

    theta_samples, psi_samples = run_mh(data)
    results[n] = {"theta": theta_samples, "psi": psi_samples}

    print(
        f"n={n:<4} | Posterior Mean Theta: {np.mean(theta_samples):.4f} | Posterior Mean Psi: {np.mean(psi_samples):.4f}"
    )
