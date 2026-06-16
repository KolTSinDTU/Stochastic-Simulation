import random as rand
import matplotlib.pyplot as plt
import numpy as np
import math


def calculate_truncated_poisson_weight(k, offered_load):
    # 1. Calculate unnormalized Poisson weight
    return (offered_load**k) / math.factorial(k)


def generate_erlang_sample(offered_load, k=1):
    # 1. Generate sample from Gamma/Erlang distribution
    return rand.gammavariate(offered_load, k)


def propose_next_state(current_state, max_capacity):
    # 1. Propose increment or decrement with equal probability
    u = rand.random()
    if u < 0.5:
        return max(0, current_state - 1)
    else:
        return min(max_capacity, current_state + 1)


def get_next_metropolis_state(current_state, offered_load, max_capacity):
    # 1. Calculate weights for current and proposed states
    current_weight = calculate_truncated_poisson_weight(current_state, offered_load)
    proposed_state = propose_next_state(current_state, max_capacity)
    proposed_weight = calculate_truncated_poisson_weight(proposed_state, offered_load)

    # 2. Calculate acceptance probability
    acceptance_ratio = min(1, proposed_weight / current_weight)

    # 3. Accept or reject based on ratio
    if acceptance_ratio == 1:
        return proposed_state
    else:
        u = rand.random()
        if u < acceptance_ratio:
            return proposed_state
        else:
            return current_state


def run_mcmc_simulation(offered_load, max_capacity):
    # 1. Initialize simulation parameters
    state_history = []
    current_state = 0
    total_steps = 3000
    burn_in = 800

    # 2. Run Markov Chain iteration
    for _ in range(total_steps):
        current_state = get_next_metropolis_state(current_state, offered_load, max_capacity)
        state_history.append(current_state)

    # 3. Return results after removing burn-in period
    return state_history[burn_in:]


def calculate_chi_squared(observed_probs, expected_probs):
    # 1. Calculate Chi-squared statistic for goodness of fit
    statistic = 0
    for i in range(len(expected_probs)):
        if expected_probs[i] > 0:
            diff = observed_probs[i] - expected_probs[i]
            statistic += (diff**2) / expected_probs[i]
    return statistic


def get_theoretical_probabilities(offered_load, max_capacity):
    # 1. Calculate normalization constant for truncated Poisson
    total_weight = sum(offered_load**i / math.factorial(i) for i in range(max_capacity + 1))
    normalization_c = 1 / total_weight
    # 2. Return normalized probabilities
    return [normalization_c * (offered_load**i) / math.factorial(i) for i in range(max_capacity + 1)]


def get_simulation_probabilities(samples, max_capacity):
    # 1. Calculate empirical distribution from samples
    total_samples = len(samples)
    return [samples.count(i) / total_samples for i in range(max_capacity + 1)]


def plot_distribution_comparison(samples, theoretical_probs, max_capacity):
    # 1. Initialize plot and figure
    fig, ax = plt.subplots(figsize=(10, 6))
    x_indices = np.arange(max_capacity + 1)
    empirical_probs = get_simulation_probabilities(samples, max_capacity)

    # 2. Plot theoretical and empirical bars
    ax.bar(
        x_indices,
        theoretical_probs,
        alpha=0.6,
        label="Theoretical Probabilities",
        color="#2c3e50",
    )

    ax.bar(
        x_indices,
        empirical_probs,
        alpha=0.6,
        label="Simulated Probabilities",
        color="#e74c3c",
    )

    # 3. Style and format the chart
    ax.set_xlabel("State (k)")
    ax.set_ylabel("Probability")
    ax.set_title("MCMC Simulation vs. Theoretical Truncated Poisson Distribution")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Setup configuration
    offered_load_val = 8
    max_capacity_val = 10
    
    # 2. Execute simulation and theoretical calculations
    simulation_results = run_mcmc_simulation(offered_load_val, max_capacity_val)
    theoretical_probs_val = get_theoretical_probabilities(offered_load_val, max_capacity_val)
    empirical_probs_val = get_simulation_probabilities(simulation_results, max_capacity_val)

    # 3. Visualize results
    plot_distribution_comparison(simulation_results, theoretical_probs_val, max_capacity_val)
    
    # 4. Display numerical results
    print(f"--- Distribution Analysis ---")
    print(f"Theoretical Probabilities: {[round(p, 4) for p in theoretical_probs_val]}")
    print(f"Simulated Probabilities: {[round(p, 4) for p in empirical_probs_val]}")
    print()
    
    chi_stat = calculate_chi_squared(empirical_probs_val, theoretical_probs_val)
    print(f"--- Statistical Tests ---")
    print(f"Chi-squared statistic: {chi_stat:.6f}")
