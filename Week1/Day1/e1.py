"""In this exercise you should implement everything including the tests
(e.g. the chi-square and KS tests) yourself. I recommend that you
also code routines for histogrammes yourself to better control
limits, but this is not strictly needed. Later, when your code is
working you are free to use builtin functions"""

"""Write a program implementing a linear congruential generator
(LCG). Be sure that the program works correctly using only
integer representation."""


"""A Linear congruential generator is defined by x_i = mod(ax_{i-1} + c, M),  
where x_0 is given and a, c and M need to be chosen carefully"""

import matplotlib.pyplot as plt
import random as rand

SAMPLES = 10_000

def lcg(x0, a, c, M):
    numbers = []
    x = x0
    for _ in range(SAMPLES):
        x = (a*x +c)%M
        numbers.append(x)

    max_x = max(numbers)
    result = [x/max_x for x in numbers]

    return result

def chai_squared(n_classes, rand_num, samples, bins):
    T = 0
    sorted_keys = sorted(rand_num.keys())
    for i in range(n_classes):
        print(f"this is the actual number of {i} smaples: {rand_num[sorted_keys[i]]}, and this is the expected number: {(samples/bins)}")
        T += (rand_num[sorted_keys[i]] - (samples/bins))**2 / (samples/bins)

    return T

def bin_list(rand_list,bins):
    size_bin = 1/bins
    sorted_list = sorted(rand_list)
    for i in range(len(rand_list)):
        if sorted_list[i]> size_bin:
            


bins = 10
seed = 3
a = 5
c = 1
M = 16

random_numbers = lcg(seed, a, c, M)
# print(random_numbers)



T = chai_squared(bins, SAMPLES, bins)
print(T)

# randints = [rand.randint for _  in range(SAMPLES)]
plt.hist(random_numbers)
# plt.scatter(range(10000), random_numbers)
# plt.scatter(range(10000), [rand.randint for _  in range(10000)])
plt.show()