import random as rand
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats

SAMPLES = 300000
BURN_IN = 8000
THINNING = 20


class proposal:
    TOGETHER = 0
    SEPARATE = 1
    GIBBS = 2


def trunc_poisson(i, j, A1, A2):
    if i + j > 10 or i < 0 or j < 0:
        return 0
    return (A1**i) * (A2**j) / (math.factorial(i) * math.factorial(j))


def propose_next_state_together(i, j, m):
    u = rand.random()
    if u < 0.25:
        return i - 1, j
    elif u < 0.5:
        return i + 1, j
    elif u < 0.75:
        return i, j - 1
    else:
        return i, j + 1


def propose_next_state(i, m):
    u = rand.random()
    if u < 0.5:
        return max(0, i - 1)
    else:
        return min(m, i + 1)


def propose_next_state_separate(i, j, m):
    u = rand.random()
    if u < 0.5:
        return propose_next_state(i, m), j
    else:
        return i, propose_next_state(j, m)


def get_gibbs_next(i, j, A1=4, A2=4, m=10):

    weights = lambda A, max_val: [
        (A**k) / math.factorial(k) for k in range(max_val + 1)
    ]

    next_i = rand.choices(range(m - j + 1), weights=weights(A1, m - j))[0]

    next_j = rand.choices(range(m - next_i + 1), weights=weights(A2, m - next_i))[0]

    return next_i, next_j


# Returns the next state and whether the move was accepted or not
def next_state(i, j, A1, A2, m, proposal_type):
    g_x = trunc_poisson(i, j, A1, A2)
    y = ()
    if proposal_type == proposal.TOGETHER:
        y = propose_next_state_together(i, j, m)

    elif proposal_type == proposal.SEPARATE:
        y = propose_next_state(i, m), propose_next_state(j, m)

    g_y = trunc_poisson(y[0], y[1], A1, A2)

    alpha = min(1, g_y / g_x)

    if alpha == 1:
        return y
    else:
        u = rand.random()
        if u < alpha:
            return y
        else:
            return i, j


def simulate(A1, A2, m, proposal_type):
    states = np.zeros(shape=(m + 1, m + 1), dtype=int)

    # Start from an arbitrary state, e.g., 0
    i = 0
    j = 0
    current_state = (i, j)

    for _ in range(SAMPLES):
        next_state_val = ()
        if proposal_type == proposal.GIBBS:
            next_state_val = get_gibbs_next(
                current_state[0], current_state[1], A1, A2, m
            )
        else:
            next_state_val = next_state(
                current_state[0], current_state[1], A1, A2, m, proposal_type
            )
        if _ >= BURN_IN and _ % THINNING == 0:
            states[current_state[0]][current_state[1]] += 1

        current_state = next_state_val

    return states


def chi_squared(samples, actual, sample_size):
    T = 0
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            diff = sample_size * samples[i][j] - sample_size * actual[i][j]
            T += (diff**2) / (sample_size * actual[i][j]) if actual[i][j] > 0 else 0

    return T


def calculate_c(A1, A2, m):
    total = 0
    for i in range(m + 1):
        for j in range(m + 1 - i):
            total += ((A1**i) * (A2**j)) / (math.factorial(i) * math.factorial(j))
    return 1 / total


def theoretical_probs(A1, A2, m):
    c = calculate_c(A1, A2, m)
    probs = np.zeros(shape=(m + 1, m + 1), dtype=float)
    for i in range(m + 1):
        for j in range(m + 1 - i):
            probs[i][j] = (
                c * ((A1**i) * (A2**j)) / (math.factorial(i) * math.factorial(j))
            )
    return probs


def simulation_probs(samples, m):
    total_samples = np.sum(samples)

    probs = np.zeros(shape=(m + 1, m + 1), dtype=float)

    for i in range(m + 1):
        for j in range(m + 1):
            probs[i][j] = (
                float(samples[i][j] / total_samples) if total_samples > 0 else 0
            )
    return probs


def plot(together_sampled, separate_sampled, gibbs_sampled, actual, m):
    # 1. Setup sample data
    x = np.arange(m + 1)  # 4 positions on X
    y = np.arange(m + 1)  # 4 positions on Y
    # Create a grid of (x, y) coordinates
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # 2. Bar dimensions
    z = np.zeros_like(x_flat)  # All bars start at Z=0
    dx = np.ones((m + 1) * (m + 1)) * 0.8  # Width of each bar
    dy = np.ones((m + 1) * (m + 1)) * 0.8  # Depth of each bar
    dz1 = together_sampled.flatten()  # Height of each bar (from samples)
    dz2 = actual.flatten()  # Height of each bar (from actual probabilities)
    dz3 = separate_sampled.flatten()  # Height of each bar (from separate samples)
    dz4 = gibbs_sampled.flatten()  # Height of each bar (from Gibbs samples)
    # 3. Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 4. Plot the bars
    ax.bar3d(
        x_flat, y_flat, z, dx, dy, dz1, color="skyblue", edgecolor="black", alpha=0.8
    )
    ax.bar3d(
        x_flat,
        y_flat,
        z,
        dx,
        dy,
        dz2,
        color="salmon",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar3d(
        x_flat,
        y_flat,
        z,
        dx,
        dy,
        dz3,
        color="lightgreen",
        edgecolor="black",
        alpha=0.8,
    )
    ax.bar3d(
        x_flat,
        y_flat,
        z,
        dx,
        dy,
        dz4,
        color="lightcoral",
        edgecolor="black",
        alpha=0.8,
    )

    # Labels and Title
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Bar Chart")
    ax.legend(
        [
            "Together Simulated Probabilities",
            "Theoretical Probabilities",
            "Separate Simulated Probabilities",
            "Gibbs Simulated Probabilities",
        ],
        loc="upper right",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    A1 = 4
    A2 = 4
    m = 10
    together_simulation = simulate(A1, A2, m, proposal.TOGETHER)
    separate_simulation = simulate(A1, A2, m, proposal.SEPARATE)
    gibbs_simulation = simulate(A1, A2, m, proposal.GIBBS)
    actual_probs = theoretical_probs(A1, A2, m)
    together_simulated_probs = simulation_probs(together_simulation, m)
    separate_simulated_probs = simulation_probs(separate_simulation, m)
    gibbs_simulated_probs = simulation_probs(gibbs_simulation, m)

    plot(
        together_simulated_probs,
        separate_simulated_probs,
        gibbs_simulated_probs,
        actual_probs,
        m,
    )
    # print(f"Actual probabilities: {actual_probs}")
    # print(f"Simulated probabilities: {simulation_probs(together_simulation, m)}")

    together_chi_statistic = chi_squared(
        together_simulated_probs, actual_probs, np.sum(together_simulation)
    )
    separate_chi_statistic = chi_squared(
        separate_simulated_probs, actual_probs, np.sum(separate_simulation)
    )
    gibbs_chi_statistic = chi_squared(
        gibbs_simulated_probs, actual_probs, np.sum(gibbs_simulation)
    )

    print(
        f"Together sampled Chi-squared statistic: {together_chi_statistic} with p-value: {1 - stats.chi2.cdf(together_chi_statistic, df=66)}"
    )
    print(
        f"Separate sampled Chi-squared statistic: {separate_chi_statistic} with p-value: {1 - stats.chi2.cdf(separate_chi_statistic, df=66)}"
    )
    print(
        f"Gibbs sampled Chi-squared statistic: {gibbs_chi_statistic} with p-value: {1 - stats.chi2.cdf(gibbs_chi_statistic, df=66)}"
    )

    # print(f"Chi-squared statistic: {chi_squared(together_simulation, actual_probs, m + 1)}")
