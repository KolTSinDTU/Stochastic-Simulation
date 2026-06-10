import math
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import statistics as stats

SAMPLES = 10_000


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

    axs.hist(pareto_values, bins=50, density=True, label=f"k={k}")
    axs.plot(x_values, pdf_values, "r-", linewidth=2)
    axs.legend()
    # axs.title(f"Pareto Distribution (beta={beta}, k={k})")
    # axs.xlabel("Value")
    # axs.ylabel("Density")
    # axs.show()


if __name__ == "__main__":
    # plot_exp_distribution()
    # plot_normal_distribution()
    fig, axs = plt.subplots(4)
    axs[0].set_title("Pareto Distribution with Different k Values")

    plot_pareto_distribution(1, 2.05, SAMPLES, axs[0])
    plot_pareto_distribution(1, 2.5, SAMPLES, axs[1])
    plot_pareto_distribution(1, 3, SAMPLES, axs[2])
    plot_pareto_distribution(1, 4, SAMPLES, axs[3])
    plt.tight_layout()
    plt.show()
