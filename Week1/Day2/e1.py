"""In this exercise, you may use built-in procedures for generating random numbers.
Compare the results obtained from simulation with the corresponding theoretical results.
Use histograms and statistical tests where appropriate.
"""

"""Choose a value for the probability parameter (p) in the geometric distribution and
simulate 10,000 observations.
"""

import random as rand
import math
import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 10_000


def generate_geometric(p_value):
    # 1. Generate a uniform random number
    u = rand.uniform(0, 1)
    
    # 2. Apply inverse transform method for geometric distribution
    return math.floor(math.log(u) / math.log(1 - p_value)) + 1


if __name__ == "__main__":
    # 1. Define probability parameters
    p_small = 0.01
    p_medium = 0.6
    p_large = 0.98

    # 2. Generate samples using NumPy for comparison
    numpy_small = [np.random.geometric(p_small) for _ in range(SAMPLES)]
    numpy_medium = [np.random.geometric(p_medium) for _ in range(SAMPLES)]
    numpy_large = [np.random.geometric(p_large) for _ in range(SAMPLES)]

    # 3. Generate samples using custom simulation
    simulated_small = [generate_geometric(p_small) for _ in range(SAMPLES)]
    simulated_medium = [generate_geometric(p_medium) for _ in range(SAMPLES)]
    simulated_large = [generate_geometric(p_large) for _ in range(SAMPLES)]

    # 4. Visualize results
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Subplot 1: Small p-value
    axs[0].hist(numpy_small, label="NumPy Geometric", alpha=0.6, color='#2c3e50')
    axs[0].hist(simulated_small, label="Simulated Geometric", alpha=0.6, color='#e74c3c')
    axs[0].set_title(f"--- Results: Small p-value (p = {p_small}) ---")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()

    # Subplot 2: Medium p-value
    axs[1].hist(numpy_medium, label="NumPy Geometric", alpha=0.6, color='#2c3e50')
    axs[1].hist(simulated_medium, label="Simulated Geometric", alpha=0.6, color='#e74c3c')
    axs[1].set_title(f"--- Results: Medium p-value (p = {p_medium}) ---")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()

    # Subplot 3: Large p-value
    axs[2].hist(numpy_large, label="NumPy Geometric", alpha=0.6, color='#2c3e50')
    axs[2].hist(simulated_large, label="Simulated Geometric", alpha=0.6, color='#e74c3c')
    axs[2].set_title(f"--- Results: Large p-value (p = {p_large}) ---")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend()

    print(f"--- Simulation Complete ---")
    print(f"Generated {SAMPLES} samples for each p-value.")

    plt.tight_layout()
    plt.show()
