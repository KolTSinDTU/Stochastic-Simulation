import math
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import scipy.stats as stats

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

# For the normal distribution, generate 100 confidence intervals for:
# the mean,
# the variance.
# Each confidence interval should be based on a sample of size n = 10, and all confidence
# intervals should have confidence level 95%.
# Present and discuss the results.

def confidence_interval(data, confidence=0.95):
    n = len(data)
    alpha = 1 - confidence
    
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    
    # Calculate margin of error using the critical value, not the confidence percentage
    margin_of_error = t_critical * (std_dev / math.sqrt(n))
    
    lower_ci = mean - margin_of_error
    upper_ci = mean + margin_of_error
    return float(lower_ci), float(upper_ci)

if __name__ == "__main__":
    # plot_exp_distribution()
    # plot_normal_distribution()
    # fig, axs = plt.subplots(4)
    # axs[0].set_title("Pareto Distribution with Different k Values")

    # plot_pareto_distribution(1, 2.05, SAMPLES, axs[0])
    # plot_pareto_distribution(1, 2.5, SAMPLES, axs[1])
    # plot_pareto_distribution(1, 3, SAMPLES, axs[2])
    # plot_pareto_distribution(1, 4, SAMPLES, axs[3])
    # plt.tight_layout()
    # plt.show()

    #plot confidence intervals for the normal distribution
    confidence_intervals = []
    for _ in range(100):
        normal_data = [box_muller()[0] for _ in range(10)]
        confidence_intervals.append(confidence_interval(normal_data))
    
    starts_box = [ci[0] for ci in confidence_intervals]
    widths_box = [ci[1] - ci[0] for ci in confidence_intervals]

    colors = []
    miss_count = 0
    for start, width in zip(starts_box, widths_box):
        end = start + width
        if start > 0 or end < 0:  # doesn't contain zero
            colors.append('red')
            miss_count += 1
        else:
            colors.append('blue')

    
    fig, axs = plt.subplots(1, figsize=(10, 6))

    axs.barh(y=range(100), width=widths_box, left=starts_box, color=colors, alpha=0.5, label='Box-Muller')
    axs.axvline(x=0, color='black', linewidth=1.5, linestyle='--', label='Zero')
    legend_elements = [
    Patch(facecolor='blue', alpha=0.5, label='Contains zero'),
    Patch(facecolor='red', alpha=0.5, label=f'Misses zero (n={miss_count})'),
    plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Zero line')]
    axs.legend(handles=legend_elements, loc='lower right')
    axs.set_xlabel('Confidence Interval Range')
    axs.set_title('Confidence Intervals for Normal Distribution')
    
    plt.tight_layout()
    plt.show()

    
