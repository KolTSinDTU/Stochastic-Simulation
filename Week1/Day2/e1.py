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
