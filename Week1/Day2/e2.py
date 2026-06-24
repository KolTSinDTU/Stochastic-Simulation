import math
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

SAMPLES = 10_000
x_probs = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
x_values = [1, 2, 3, 4, 5, 6]

# --- 1. Direct Method ---
def direct_method(x):
    u = rand.uniform(0, 1)
    curr = 0
    for i in range(len(x)):
        prev = curr
        curr += x[i]
        if prev < u <= curr:
            return i + 1
    return len(x)

# --- 2. Rejection Method ---
def rejection(x):
    c = max(x)  # optimal bounding constant
    while True:
        u1 = rand.uniform(0, 1)
        u2 = rand.uniform(0, 1)
        I = math.floor(len(x) * u1) + 1  # Propose uniform integer 1..6
        if x[I-1]/c >= u2:               # Accept/Reject condition
            return I

# --- 3. Alias Method ---
def setup_alias(probabilities):
    """ O(n) setup to precompute prob and alias tables """
    n = len(probabilities)
    alias_table = [0] * n
    prob = [0.0] * n
    scaled_prob = [p * n for p in probabilities]

    small, large = [], []
    for i, sp in enumerate(scaled_prob):
        if sp < 1.0:
            small.append(i)
        else:
            large.append(i)
            
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled_prob[s]
        alias_table[s] = l
        scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - 1.0
        
        if scaled_prob[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
            
    for i in large + small:
        prob[i] = 1.0
        
    return prob, alias_table

def sample_alias(prob_table, alias_table):
    """ O(1) sampling operation """
    n = len(prob_table)
    i = rand.randint(0, n-1)
    u = rand.uniform(0, 1)
    if u < prob_table[i]:
        return i + 1
    else:
        return alias_table[i] + 1

# --- Generation ---
# Precompute alias tables ONCE
prob_tbl, alias_tbl = setup_alias(x_probs)

direct_values = [direct_method(x_probs) for _ in range(SAMPLES)]
rejection_values = [rejection(x_probs) for _ in range(SAMPLES)]
alias_values = [sample_alias(prob_tbl, alias_tbl) for _ in range(SAMPLES)]

# --- Statistical Testing & Plotting ---
expected_counts = [SAMPLES * p for p in x_probs]
methods = {
    "Direct": direct_values,
    "Rejection": rejection_values,
    "Alias": alias_values
}

# Plotting the histograms - add good titles and labels for clarity
fig, axs = plt.subplots(3)
axs[0].hist(direct_values, label="Direct Method", alpha=0.6)
axs[1].hist(rejection_values, label="Rejection Method", alpha=0.6)
axs[2].hist(alias_values, label="Alias Method", alpha=0.6)
axs[0].legend() 
axs[1].legend() 
axs[2].legend() 
axs[0].set_title("Comparisons of Sampling Methods")
plt.tight_layout()
plt.show()

print("\n--- Chi-Square Goodness-of-Fit Results")
for name, data in methods.items():
    counts = [data.count(i) for i in x_values]
    chi_stat, p_val = chisquare(f_obs=counts, f_exp=expected_counts)
    print(f"{name.ljust(10)}: Stat = {chi_stat:.4f}, p-value = {p_val:.4f}")