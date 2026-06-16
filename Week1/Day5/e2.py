import random as rand
import matplotlib.pyplot as plt
import numpy as np
import math

TOTAL_SAMPLES = 3000
BURN_IN_PERIOD = 800


class ProposalMethod:
    TOGETHER = 0
    SEPARATE = 1
    GIBBS = 2


def calculate_joint_truncated_poisson(i, j, load_1, load_2):
    # 1. Calculate joint weight for independent Poisson variables
    return (load_1**i) * (load_2**j) / (math.factorial(i) * math.factorial(j))


def propose_next_state_together(i, j, max_capacity):
    # 1. Propose change to both variables simultaneously
    u = rand.random()
    if u < 0.25:
        return max(0, i - 1), max(0, j - 1)
    elif u < 0.5:
        return min(max_capacity, i + 1), min(max_capacity, j + 1)
    elif u < 0.75:
        return max(0, i - 1), min(max_capacity, j + 1)
    else:
        return min(max_capacity, i + 1), max(0, j - 1)


def propose_individual_step(current_val, max_capacity):
    # 1. Propose a step for a single variable
    u = rand.random()
    if u < 0.5:
        return max(0, current_val - 1)
    else:
        return min(max_capacity, current_val + 1)


def get_gibbs_sample(i, j, load_1, load_2, max_capacity):
    # 1. Helper to generate weights for discrete sampling
    def get_weights(load, limit):
        return [(load**k) / math.factorial(k) for k in range(limit + 1)]

    # 2. Sample next i given j
    next_i = rand.choices(range(max_capacity - j + 1), weights=get_weights(load_1, max_capacity - j))[0]

    # 3. Sample next j given i
    next_j = rand.choices(range(max_capacity - next_i + 1), weights=get_weights(load_2, max_capacity - next_i))[0]

    return next_i, next_j


def get_next_mcmc_state(i, j, load_1, load_2, max_capacity, method):
    # 1. Calculate current weight
    current_weight = calculate_joint_truncated_poisson(i, j, load_1, load_2)
    
    # 2. Propose next state based on method
    if method == ProposalMethod.TOGETHER:
        proposed_state = propose_next_state_together(i, j, max_capacity)
        while proposed_state[0] + proposed_state[1] > max_capacity:
            proposed_state = propose_next_state_together(i, j, max_capacity)
    elif method == ProposalMethod.SEPARATE:
        proposed_state = propose_individual_step(i, max_capacity), propose_individual_step(j, max_capacity)
        while proposed_state[0] + proposed_state[1] > max_capacity:
            proposed_state = propose_next_state_together(i, j, max_capacity)

    # 3. Calculate acceptance ratio
    proposed_weight = calculate_joint_truncated_poisson(proposed_state[0], proposed_state[1], load_1, load_2)
    acceptance_ratio = min(1, proposed_weight / current_weight)

    # 4. Metropolis acceptance step
    if acceptance_ratio == 1:
        return proposed_state
    else:
        u = rand.random()
        if u < acceptance_ratio:
            return proposed_state
        else:
            return i, j


def run_simulation(load_1, load_2, max_capacity, method):
    # 1. Initialize state space and chain
    frequency_matrix = np.zeros(shape=(max_capacity + 1, max_capacity + 1), dtype=int)
    current_i, current_j = 0, 0

    # 2. Run Markov Chain
    for step in range(TOTAL_SAMPLES):
        if method == ProposalMethod.GIBBS:
            next_i, next_j = get_gibbs_sample(current_i, current_j, load_1, load_2, max_capacity)
        else:
            next_i, next_j = get_next_mcmc_state(current_i, current_j, load_1, load_2, max_capacity, method)
        
        # 3. Record frequency after burn-in
        if step >= BURN_IN_PERIOD:
            frequency_matrix[current_i][current_j] += 1

        current_i, current_j = next_i, next_j

    return frequency_matrix


def calculate_chi_squared(observed_probs, expected_probs):
    # 1. Compute chi-squared goodness of fit
    chi_stat = 0
    for i in range(observed_probs.shape[0]):
        for j in range(observed_probs.shape[1]):
            if expected_probs[i][j] > 0:
                diff = observed_probs[i][j] - expected_probs[i][j]
                chi_stat += (diff**2) / expected_probs[i][j]
    return chi_stat


def get_theoretical_probabilities(load_1, load_2, max_capacity):
    # 1. Calculate normalization constant
    total_weight = 0
    for i in range(max_capacity + 1):
        for j in range(max_capacity + 1 - i):
            total_weight += ((load_1**i) * (load_2**j)) / (math.factorial(i) * math.factorial(j))
    normalization_c = 1 / total_weight
    
    # 2. Fill probability matrix
    probs_matrix = np.zeros(shape=(max_capacity + 1, max_capacity + 1), dtype=float)
    for i in range(max_capacity + 1):
        for j in range(max_capacity + 1 - i):
            probs_matrix[i][j] = normalization_c * ((load_1**i) * (load_2**j)) / (math.factorial(i) * math.factorial(j))
    return probs_matrix


def get_empirical_probabilities(frequency_matrix, max_capacity):
    # 1. Convert counts to probabilities
    total_samples_count = TOTAL_SAMPLES - BURN_IN_PERIOD
    return frequency_matrix.astype(float) / total_samples_count


def plot_results(together_probs, separate_probs, gibbs_probs, theoretical_probs, max_capacity):
    # 1. Prepare grid data
    x_coords = np.arange(max_capacity + 1)
    y_coords = np.arange(max_capacity + 1)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # 2. Dimensions and heights
    z_base = np.zeros_like(x_flat)
    width = depth = 0.7
    
    # 3. Create figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 4. Plot bars with hex colors
    ax.bar3d(x_flat, y_flat, z_base, width, depth, together_probs.flatten(), color="#e74c3c", alpha=0.5, label="Together")
    ax.bar3d(x_flat, y_flat, z_base, width, depth, theoretical_probs.flatten(), color="#2c3e50", alpha=0.3, label="Theoretical")
    
    # 5. Styling
    ax.set_xlabel("State i")
    ax.set_ylabel("State j")
    ax.set_zlabel("Probability")
    ax.set_title("3D Distribution Comparison")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Configuration
    load_val_1 = 4
    load_val_2 = 4
    max_cap_val = 10
    
    # 2. Run simulations
    together_freq = run_simulation(load_val_1, load_val_2, max_cap_val, ProposalMethod.TOGETHER)
    separate_freq = run_simulation(load_val_1, load_val_2, max_cap_val, ProposalMethod.SEPARATE)
    gibbs_freq = run_simulation(load_val_1, load_val_2, max_cap_val, ProposalMethod.GIBBS)
    
    # 3. Calculate probabilities
    theory_probs = get_theoretical_probabilities(load_val_1, load_val_2, max_cap_val)
    together_probs_val = get_empirical_probabilities(together_freq, max_cap_val)
    separate_probs_val = get_empirical_probabilities(separate_freq, max_cap_val)
    gibbs_probs_val = get_empirical_probabilities(gibbs_freq, max_cap_val)

    # 4. Visualize
    plot_results(together_probs_val, separate_probs_val, gibbs_probs_val, theory_probs, max_cap_val)

    # 5. Output statistics
    print(f"--- MCMC Statistical Analysis ---")
    print(f"Together Proposal Chi-squared: {calculate_chi_squared(together_probs_val, theory_probs):.6f}")
    print(f"Separate Proposal Chi-squared: {calculate_chi_squared(separate_probs_val, theory_probs):.6f}")
    print(f"Gibbs Sampling Chi-squared: {calculate_chi_squared(gibbs_probs_val, theory_probs):.6f}")
