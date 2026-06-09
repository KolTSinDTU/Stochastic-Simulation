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

def alias(x):
    pass

x = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]

direct_values = [direct_method(x) for _ in range(SAMPLES)]
rejection_values = [rejection(x) for _ in range(SAMPLES)]
# alias_values = [alias(x) for _ in range(SAMPLES)]

fig, axs = plt.subplots(2)
axs[0].hist(direct_values)

axs[1].hist(rejection_values)

# axs[2].hist(alias_values)
plt.show()
