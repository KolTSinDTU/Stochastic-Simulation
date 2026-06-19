import numpy as np
from scipy.linalg import expm
from scipy.stats import kstest

Q_s = np.array([
    [-0.0085, 0.005,  0.0025, 0     ],
    [ 0,     -0.014,  0.005,  0.004 ],
    [ 0,      0,     -0.008,  0.003 ],
    [ 0,      0,      0,     -0.009 ]
])
p_0 = np.array([1, 0, 0, 0])
ones = np.ones(4)

Q = np.zeros((5, 5))
Q[:4, :4] = Q_s
Q[0, 4], Q[1, 4], Q[2, 4], Q[3, 4] = 0.001, 0.005, 0.005, 0.009

# Run Simulation (1000 women)
n_women = 1000
lifetimes = np.zeros(n_women)

for i in range(n_women):
    state = 0
    current_time = 0.0
    while state != 4:
        lambda_i = -Q[state, state]
        current_time += np.random.exponential(1 / lambda_i)
        
        probs = Q[state, :].copy()
        probs[state] = 0
        probs = probs / lambda_i
        state = np.random.choice(5, p=probs)
        
    lifetimes[i] = current_time

# 4. Define Bins and Calculate Observed Counts
# Use deciles (10 bins) to ensure enough samples per bin (expected ~100 per bin)
percentiles = np.linspace(0, 100, 11)
bins = np.percentile(lifetimes, percentiles)
bins[0] = 0.0          
bins[-1] = np.inf      

from scipy.stats import chisquare

observed_counts, _ = np.histogram(lifetimes, bins=bins)

expected_probs = np.zeros(len(bins) - 1)

for i in range(len(bins) - 1):
    t_start = bins[i]
    t_end = bins[i+1]
    
    # Calculate CDF at start of bin
    if t_start == 0.0:
        cdf_start = 0.0
    else:
        cdf_start = 1.0 - (p_0 @ expm(Q_s * t_start) @ ones)
        
    # Calculate CDF at end of bin
    if t_end == np.inf:
        cdf_end = 1.0
    else:
        cdf_end = 1.0 - (p_0 @ expm(Q_s * t_end) @ ones)
        
    # Probability of falling in the bin is F(end) - F(start)
    expected_probs[i] = cdf_end - cdf_start

expected_counts = expected_probs * n_women

# Perform the Chi-Square Test
chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

print("Chi-Square Goodness-of-Fit Test Results:")
print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"P-value:              {p_value:.4f}\n")

if p_value > 0.05:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("The binned simulated lifetimes match the expected theoretical frequencies.")
else:
    print("Conclusion: Reject the null hypothesis.")
    print("The simulated empirical lifetime distribution deviates significantly from the theoretical model.")