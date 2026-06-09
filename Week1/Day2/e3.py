import math
import matplotlib.pyplot as plt
import random as rand
import numpy as np

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


def plot_pareto_distribution(beta: float = 1, k: float = 2, samples: int = SAMPLES):
    pareto_values = [pareto_distribution(beta, k) for _ in range(samples)]

    x_values = np.linspace(beta, max(pareto_values), 1000)
    pdf_values = (k * (beta**k)) / (x_values ** (k + 1))

    plt.hist(pareto_values, bins=50, density=True)
    plt.plot(x_values, pdf_values, "r-", linewidth=2)
    plt.title(f"Pareto Distribution (beta={beta}, k={k})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


if __name__ == "__main__":
    # plot_exp_distribution()
    # plot_normal_distribution()
    plot_pareto_distribution()
