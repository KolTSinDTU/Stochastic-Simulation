import random as rand
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats


def trunc_poisson(i, A):
    return (A**i) / math.factorial(i)


def erling(A, k=1):
    return rand.gammavariate(A, k)
    # return stats.erlang.rvs([k], scale=1 / A)


def propose_next_state(i, m):
    u = rand.random()
    if u < 0.5:
        return max(0, i - 1)
    else:
        return min(m, i + 1)


# Returns the next state and whether the move was accepted or not
def next_state(x, A, m):
    g_x = trunc_poisson(x, A)
    y = propose_next_state(x, m)
    g_y = trunc_poisson(y, A)

    alpha = min(1, g_y / g_x)

    if alpha == 1:
        return y
    else:
        u = rand.random()
        if u < alpha:
            return y
        else:
            return x


def simulate(A, m):
    states = []

    # Start from an arbitrary state, e.g., 0
    current_state = 0

    for _ in range(3000):
        next_state_val = next_state(current_state, A, m)
        states.append(next_state_val)
        current_state = next_state_val

    return states[800:]


def chi_squared(samples, actual, sample_size):
    T = 0
    for i in range(len(actual)):
        observed = sample_size * samples[i]
        expected = sample_size * actual[i]
        diff = observed - expected
        T += (diff**2) / expected if expected > 0 else 0

    return T


def calculate_c(A, m):
    total = sum(A**i / math.factorial(i) for i in range(m + 1))
    return 1 / total


def theoretical_probs(A, m):
    c = calculate_c(A, m)
    return [c * (A**i) / math.factorial(i) for i in range(m + 1)]


def simulation_probs(samples, m):
    total_samples = len(samples)
    return [samples.count(i) / total_samples for i in range(m + 1)]


def plot(samples, actual, m):
    plt.bar(
        range(m + 1),
        actual,
        alpha=0.5,
        label="Theoretical Probabilities",
        color="blue",
    )

    plt.bar(
        range(m + 1),
        simulation_probs(samples, m),
        alpha=0.5,
        label="Simulated Probabilities",
        color="red",
    )

    # Formatting the plot
    plt.xlim(left=0)  # Erlang distribution is only defined for x >= 0
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Comparison of sampled and theoretical distributions")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    A = 8
    m = 10
    simulations = simulate(A, m)
    actual_probs = theoretical_probs(A, m)

    # plot(simulations, actual_probs, m)
    print(f"Actual probabilities: {actual_probs}")
    print(f"Simulated probabilities: {simulation_probs(simulations, m)}")

    chi_statistic = chi_squared(
        simulation_probs(simulations, m), actual_probs, len(simulations)
    )

    print(f"Chi-squared statistic: {chi_statistic}")

    print(f"p-value : {stats.chi2.sf(chi_statistic, df=m)}")
