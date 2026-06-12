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
    cur = i / m
    val = math.fabs(rand.gauss(cur, 1))
    return int(val * m)


# Returns the next state and whether the move was accepted or not
def next_state(x, A, m):
    g_x = trunc_poisson(x, A)
    y = propose_next_state(x, m)
    g_y = trunc_poisson(y, A)

    alpha = min(1, g_y / g_x)

    if alpha == 1:
        return y, True
    else:
        u = rand.random()
        if u < alpha:
            return y, True
        else:
            return x, False


def simulate(A, m):
    states = []

    # Start from an arbitrary state, e.g., 0
    current_state = 0

    for _ in range(3000):
        next_state_val, accepted = next_state(current_state, A, m)
        if next_state_val != current_state and accepted:
            states.append(next_state_val)
            current_state = next_state_val

    return states[800:]


def chi_squared(samples, actual, bins):
    T = 0
    n = len(samples)

    observed = [0] * bins
    expected = [0] * bins
    for u in actual:
        bin_index = int(u * bins)
        expected[bin_index] += 1
    for u in samples:
        bin_index = int(u * bins)
        observed[bin_index] += 1

    for i in range(bins):
        T += ((observed[i] - expected[i]) ** 2) / expected[i]

    return T


def plot(samples, actual):
    plt.hist(
        samples,
        bins=60,
        range=(0, 20),
        alpha=0.6,
        color="royalblue",
        edgecolor="blue",
        label=f"samples",
    )

    plt.hist(
        actual,
        bins=60,
        range=(0, 20),
        alpha=0.6,
        color="darkorange",
        edgecolor="chocolate",
        label=f"actual",
    )

    # Formatting the plot
    plt.xlim(left=0)  # Erlang distribution is only defined for x >= 0
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.title("Comparison of Two Erlang Probability Distributions")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    A = 8
    m = 10
    simulations = simulate(A, m)
    actual_probs = [erling(A) for _ in range(len(simulations))]

    plot(simulations, actual_probs)

    # print(f"Chi-squared statistic: {chi_squared(simulations, actual_probs, m + 1)}")
