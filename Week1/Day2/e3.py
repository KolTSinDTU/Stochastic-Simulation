import math
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import statistics as stats

SAMPLES = 100


def exp_distribution(lam):
    u = rand.random()
    return -math.log(u) / lam


def box_muller():
    u1 = rand.random()
    u2 = rand.random()

    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)

    return z0, z1


def pareto_distribution(beta, k):
    u = rand.random()
    return beta * (u ** (-1 / k))


def pareto_composition(beta):
    e1 = exp_distribution(beta)
    e2 = exp_distribution(e1)
    return e2


def plot_exp_distribution(lam: float = 10, samples: int = SAMPLES):
    exp_values = [exp_distribution(lam) for _ in range(samples)]

    x_values = np.linspace(0, max(exp_values), 1000)
    pdf_values = lam * np.exp(-lam * x_values)

    plt.hist(exp_values, bins=50, density=True)
    plt.plot(x_values, pdf_values, "r-", linewidth=2)
    plt.title(f"Exponential Distribution (lambda={lam})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def plot_normal_distribution(samples: int = SAMPLES):
    normal_values = [box_muller()[0] for _ in range(samples)]

    x_values = np.linspace(min(normal_values), max(normal_values), 1000)
    pdf_values = (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * x_values**2)

    plt.hist(normal_values, bins=50, density=True)
    plt.plot(x_values, pdf_values, "r-", linewidth=2)
    plt.title("Standard Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def plot_pareto_distribution(
    beta: float = 1, k: float = 2, samples: int = SAMPLES, axs=None
):
    pareto_values = [pareto_distribution(beta, k) for _ in range(samples)]

    x_values = np.linspace(beta, max(pareto_values), 1000)
    pdf_values = (k * (beta**k)) / (x_values ** (k + 1))

    sampled_mean = stats.mean(pareto_values)
    theoretical_mean = (k * beta) / (k - 1)

    sampled_variance = stats.variance(pareto_values)
    theoretical_variance = (k * (beta**2)) / ((k - 1) ** 2 * (k - 2))

    print(f"Sampled Mean: {sampled_mean:.4f}, Theoretical Mean: {theoretical_mean:.4f}")
    print(
        f"Sampled Variance: {sampled_variance:.4f}, Theoretical Variance: {theoretical_variance:.4f}"
    )

    axs.hist(pareto_values, bins=50, density=True, label=f"k={k}")
    axs.plot(x_values, pdf_values, "r-", linewidth=2)
    axs.legend()
    # axs.title(f"Pareto Distribution (beta={beta}, k={k})")
    # axs.xlabel("Value")
    # axs.ylabel("Density")
    # axs.show()


def plot_pareto_composition():
    beta = 1
    k = 1
    pareto_values_composition = [pareto_composition(beta) for _ in range(SAMPLES)]
    pareto_values_direct = [pareto_distribution(beta, k) for _ in range(SAMPLES)]

    plt.figure(figsize=(8, 5))

    # Plot just the two histograms with transparency (alpha) and labels
    plt.hist(pareto_values_composition, bins=50, density=True, alpha=0.6, label="Composition Method")
    plt.hist(pareto_values_direct, bins=50, density=True, alpha=0.6, label="Inverse Method")

    # Start the plot from x > beta
    plt.xlim(left=beta)

    # Add appropriate legends
    plt.legend()

    plt.title("Pareto Distribution: Composition vs Inverse Method")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


if __name__ == "__main__":
    # plot_exp_distribution()
    # plot_normal_distribution()
    # fig, axs = plt.subplots(1)
    # # axs[0].set_title("Pareto Distribution with Different k Values")

    # # plot_pareto_distribution(1, 2.05, SAMPLES, axs[0])
    # # plot_pareto_distribution(1, 2.5, SAMPLES, axs[1])
    # plot_pareto_distribution(1, 3, SAMPLES, axs)
    # # plot_pareto_distribution(1, 4, SAMPLES, axs[3])
    # plt.tight_layout()
    # plt.show()

    plot_pareto_composition()
