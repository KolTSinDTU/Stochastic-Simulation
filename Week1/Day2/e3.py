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


def test_exponential_fit(sample_data, lambda_param):
    alpha = 0.05  # Significance level
    # Convert rate parameter to scale parameter for scipy
    scale_param = 1 / lambda_param

    # Perform the KS test against the theoretical exponential distribution
    ks_statistic, p_value = stats.kstest(sample_data, "expon", args=(0, scale_param))

    return {"ks_statistic": ks_statistic, "p_value": p_value}


def test_normal_fit(sample_data):

    # Perform the KS test against the theoretical normal distribution
    ks_statistic, p_value = stats.kstest(sample_data, "norm", args=(0, 1))

    return ks_statistic, p_value


def test_pareto_fit(sample_data, k, beta):

    # Perform the KS test against the theoretical 'pareto' distribution
    ks_statistic, p_value = stats.kstest(sample_data, "pareto", args=(k, 0, beta))

    return ks_statistic, p_value


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


def theoretical_pareto_mean(beta, k):
    if k <= 1:
        raise ValueError("Mean is undefined for k <= 1.")
    return (k * beta) / (k - 1)


def theoretical_pareto_variance(beta, k):
    if k <= 2:
        raise ValueError("Variance is undefined for k <= 2.")
    return (k * (beta**2)) / ((k - 1) ** 2 * (k - 2))


def compare_pareto_statistics(beta, k):
    sampled_values = [pareto_distribution(beta, k) for _ in range(SAMPLES)]
    sampled_mean = np.mean(sampled_values)
    sampled_variance = np.var(sampled_values, ddof=1)

    theoretical_mean = theoretical_pareto_mean(beta, k)
    theoretical_variance = theoretical_pareto_variance(beta, k)
    print(f"For Pareto Distribution with beta={beta} and k={k}:")
    print(f"Sampled Mean: {sampled_mean:.4f}, Theoretical Mean: {theoretical_mean:.4f}")
    print(
        f"Sampled Variance: {sampled_variance:.4f}, Theoretical Variance: {theoretical_variance:.4f} \n"
    )


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

    comparison_result = test_exponential_fit(exp_values, lam)
    print(
        f"KS Statistic: {comparison_result['ks_statistic']:.4f}, P-value: {comparison_result['p_value']:.4f}"
    )


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

    ks_statistic, p_value = test_normal_fit(normal_values)
    print(f"KS Statistic: {ks_statistic:.4f}, P-value: {p_value:.4f}")


def plot_pareto_distribution(
    beta: float = 1, k: float = 2, samples: int = SAMPLES, axs=None
):
    pareto_values = [pareto_distribution(beta, k) for _ in range(samples)]

    x_values = np.linspace(beta, max(pareto_values), 1000)
    pdf_values = (k * (beta**k)) / (x_values ** (k + 1))

    sampled_mean = np.mean(pareto_values)
    theoretical_mean = (k * beta) / (k - 1)

    sampled_variance = np.var(pareto_values, ddof=1)
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
    ks_statistic, p_value = test_pareto_fit(pareto_values, k, beta)
    print(f"KS Statistic: {ks_statistic:.4f}, P-value: {p_value:.4f}")


# For the normal distribution, generate 100 confidence intervals for:
# the mean,
# the variance.
# Each confidence interval should be based on a sample of size n = 10, and all confidence
# intervals should have confidence level 95%.
# Present and discuss the results.


def confidence_interval(data, variance_ci=False, confidence=0.95):
    n = len(data)
    alpha = 1 - confidence
    df = n - 1

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # Calculate margin of error using the critical value, not the confidence percentage
    margin_of_error = t_critical * (std_dev / math.sqrt(n))

    mean_lower_ci = mean - margin_of_error
    mean_upper_ci = mean + margin_of_error

    sample_var = np.var(data, ddof=1)

    chi2_lower_crit = stats.chi2.ppf(alpha / 2, df=df)
    chi2_upper_crit = stats.chi2.ppf(1 - alpha / 2, df=df)

    var_lower = (df * sample_var) / chi2_upper_crit
    var_upper = (df * sample_var) / chi2_lower_crit

    if variance_ci:
        return float(var_lower), float(var_upper)
    return float(mean_lower_ci), float(mean_upper_ci)


def plot_pareto_composition():
    beta = 1
    k = 1
    pareto_values_composition = [pareto_composition(beta) for _ in range(SAMPLES)]
    pareto_values_direct = [pareto_distribution(beta, k) for _ in range(SAMPLES)]

    plt.figure(figsize=(8, 5))

    # Plot just the two histograms with transparency (alpha) and labels
    plt.hist(
        pareto_values_composition,
        bins=50,
        density=True,
        alpha=0.6,
        label="Composition Method",
    )
    plt.hist(
        pareto_values_direct, bins=50, density=True, alpha=0.6, label="Inverse Method"
    )

    # Start the plot from x > beta
    plt.xlim(left=beta)

    # Add appropriate legends
    plt.legend()

    plt.title("Pareto Distribution: Composition vs Inverse Method")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def plot_ci(confidence_intervals, theoretical_value=1):
    starts_box = [ci[0] for ci in confidence_intervals]
    widths_box = [ci[1] - ci[0] for ci in confidence_intervals]

    colors = []
    miss_count = 0
    for start, width in zip(starts_box, widths_box):
        end = start + width
        if (
            start > theoretical_value or end < theoretical_value
        ):  # doesn't contain the theoretical value
            colors.append("red")
            miss_count += 1
        else:
            colors.append("blue")

    fig, axs = plt.subplots(1, figsize=(10, 6))

    axs.barh(
        y=range(100),
        width=widths_box,
        left=starts_box,
        color=colors,
        alpha=0.5,
        label="Box-Muller",
    )
    axs.axvline(
        x=theoretical_value,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Theoretical Value",
    )
    legend_elements = [
        Patch(facecolor="blue", alpha=0.5, label=f"Contains {theoretical_value}"),
        Patch(
            facecolor="red",
            alpha=0.5,
            label=f"Misses {theoretical_value} (n={miss_count})",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="Theoretical Value line",
        ),
    ]
    axs.legend(handles=legend_elements, loc="lower right")
    axs.set_xlabel("Confidence Interval Range")
    axs.set_title("Confidence Intervals for Normal Distribution")

    plt.tight_layout()
    plt.show()


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
    # compare_pareto_statistics(1, 2.05)
    # compare_pareto_statistics(1, 2.5)
    # compare_pareto_statistics(1, 3)
    # compare_pareto_statistics(1, 4)

    # confidence intervals for the normal distribution
    confidence_intervals = []
    for _ in range(100):
        normal_data = [box_muller()[0] for _ in range(10)]
        confidence_intervals.append(confidence_interval(normal_data))
    plot_ci(
        confidence_intervals, theoretical_value=0
    )  # Theoretical mean of standard normal is 0

    confidence_intervals_var = []
    for _ in range(100):
        normal_data = [box_muller()[0] for _ in range(10)]
        confidence_intervals_var.append(
            confidence_interval(normal_data, variance_ci=True)
        )
    plot_ci(
        confidence_intervals_var, theoretical_value=1
    )  # Theoretical variance of standard normal is 1
