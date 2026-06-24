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
from scipy.stats import chisquare, geom

SAMPLES = 10_000


def generate_geom(p):
    u = rand.uniform(0, 1)
    return math.floor(math.log(u) / math.log(1 - p)) + 1


small = 0.01
medium = 0.6
large = 0.98


import numpy as np
import matplotlib.pyplot as plt

# Assuming SAMPLES, small, medium, large, and generate_geom are defined above

small_geom = [np.random.geometric(small) for _ in range(SAMPLES)]
medium_geom = [np.random.geometric(medium) for _ in range(SAMPLES)]
large_geom = [np.random.geometric(large) for _ in range(SAMPLES)]

small_sim_geom = [generate_geom(small) for _ in range(SAMPLES)]
medium_sim_geom = [generate_geom(medium) for _ in range(SAMPLES)]
large_sim_geom = [generate_geom(large) for _ in range(SAMPLES)]

fig, axs = plt.subplots(3)

# Subplot 1
axs[0].hist(small_geom, label="Numpy Geometric", alpha=0.6)
axs[0].hist(small_sim_geom, label="Simulated Geometric", alpha=0.6)
axs[0].set_title(f"Small p-value (p = {small})")
axs[0].legend()

# Subplot 2
axs[1].hist(medium_geom, label="Numpy Geometric", alpha=0.6)
axs[1].hist(medium_sim_geom, label="Simulated Geometric", alpha=0.6)
axs[1].set_title(f"Medium p-value (p = {medium})")
axs[1].legend()

# Subplot 3
axs[2].hist(large_geom, label="Numpy Geometric", alpha=0.6)
axs[2].hist(large_sim_geom, label="Simulated Geometric", alpha=0.6)
axs[2].set_title(f"Large p-value (p = {large})")
axs[2].legend()

# Optional but recommended: layout adjustment to prevent overlapping text
plt.tight_layout()
plt.show()

def gof(data, p, max_k):
    # Calculate Observed Frequencies (with tail pooling)
    observed_counts = np.zeros(max_k)
    for val in data:
        if val >= max_k:
            observed_counts[max_k - 1] += 1  # Pool into the last bin
        else:
            observed_counts[val - 1] += 1    # Standard bins (0-indexed array)

    # Calculate Expected Frequencies using scipy.stats.geom
    expected_probs = np.zeros(max_k)

    # Get the exact Probability Mass Function (PMF) for standard bins
    for k in range(1, max_k):
        expected_probs[k - 1] = geom.pmf(k, p)

    # Get the Survival Function (SF) for the pooled tail
    expected_probs[max_k - 1] = geom.sf(max_k - 1, p)   

    # Convert probabilities to expected counts
    expected_counts = expected_probs * len(data)

    # Run the Chi-Square Test
    chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    return chi2_stat, p_value

#gof tests for small, medium, and large p-values
max_k = 12  # Define a maximum value for pooling
chi2_small, p_small = gof(small_sim_geom, small, max_k)
chi2_medium, p_medium = gof(medium_sim_geom, medium, max_k)
chi2_large, p_large = gof(large_sim_geom, large, max_k)

print(f"Small p-value (p = {small}):")
print(f"Chi-Square Statistic: {chi2_small}, p-value: {p_small}")
print(f"Medium p-value (p = {medium}):")
print(f"Chi-Square Statistic: {chi2_medium}, p-value: {p_medium}")
print(f"Large p-value (p = {large}):")
print(f"Chi-Square Statistic: {chi2_large}, p-value: {p_large}")