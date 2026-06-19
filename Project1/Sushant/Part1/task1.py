import numpy as np
import matplotlib.pyplot as plt


def p_matrix():
    p = np.zeros((5, 5), dtype=float)

    p[0] = [0.9915, 0.005, 0.0025, 0, 0.001]
    p[1] = [0, 0.986, 0.005, 0.004, 0.005]
    p[2] = [0, 0, 0.992, 0.003, 0.005]
    p[3] = [0, 0, 0, 0.991, 0.009]
    p[4] = [0, 0, 0, 0, 1.0]
    return p

P = p_matrix()
n_women = 1000
lifetimes = []
local_recurrence_count = 0

for _ in range(n_women):
    state = 0  # Starts in State 1 (0-indexed)
    months = 0
    had_local = False
    
    while state != 4:  # Loop until State 5 (Death, index 4)
        # Sample next state based on current state's probability row
        state = np.random.choice(5, p=P[state])
        months += 1
        
        # Check if State 2 (Local recurrence, index 1) is entered
        if state == 1 or state == 3:
            had_local = True
            
    lifetimes.append(months)
    if had_local:
        local_recurrence_count += 1

proportion_local = local_recurrence_count / n_women
print(f"Proportion of women with local recurrence: {proportion_local:.4f}")

# Summarize lifetime distribution
plt.hist(lifetimes, bins=40, edgecolor='black', alpha=0.7)
plt.title("Lifetime Distribution After Surgery")
plt.xlabel("Months")
plt.ylabel("Number of Women")
plt.grid(axis='y', alpha=0.5)
plt.show()