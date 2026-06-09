import random as rand
import matplotlib.pyplot as plt
import math

SAMPLES = 10_000

def direct_method(x):
    u = rand.uniform(0,1)
    curr = 0
    for i in range(len(x)):
        prev = curr
        curr  += x[i]
        if u > prev and u <= curr:
            return i + 1

def rejection(x):
    c = max(x) + 0.001
    u1 = rand.uniform(0,1)
    u2 = rand.uniform(0,1)
    I = math.floor(len(x) * u1) + 1
    if x[I-1]/c >= u2:
        return I
    while (x[I-1]/c < u2):
        u1 = rand.uniform(0,1)
        u2 = rand.uniform(0,1)
        I = math.floor(len(x) * u1) +1

        if x[I-1]/c >= u2:
            return I

def alias(probabilities):
    n = len(probabilities)

    alias_table = [0] * n
    prob = [0] * n
    scaled_prob = [p * n for p in probabilities]

    small = []
    large = []  
    for i, sp in enumerate(scaled_prob):
        if sp < 1:
            small.append(i)
        else:
            large.append(i)
    #robin hood loop 
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled_prob[s]
        alias_table[s] = l
        scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - 1      
        if scaled_prob[l] < 1:
            small.append(l)
        else:
            large.append(l) 
    for i in large + small:
        prob[i] = 1.0

    
    i = rand.randint(0, n-1)
    u = rand.uniform(0, 1)
    if u < prob[i]:
        return i + 1
    else:
        return alias_table[i] + 1  

x = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]

direct_values = [direct_method(x) for _ in range(SAMPLES)]
rejection_values = [rejection(x) for _ in range(SAMPLES)]
alias_values = [alias(x) for _ in range(SAMPLES)]
# alias_values = [alias(x) for _ in range(SAMPLES)]

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
