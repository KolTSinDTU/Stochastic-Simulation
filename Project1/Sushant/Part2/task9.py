import numpy as np
import matplotlib.pyplot as plt

# Control Group Matrix (from previous task)
Q_control = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [ 0,     -0.014,  0.005,  0.004, 0.005],
    [ 0,      0,     -0.008,  0.003, 0.005],
    [ 0,      0,      0,     -0.009, 0.009],
    [ 0,      0,      0,      0,     0    ]
])

# Treatment Group Matrix
Q_treatment = np.array([
    [-0.00475, 0.0025, 0.00125, 0,     0.001],
    [ 0,      -0.007,  0,       0.002, 0.005],
    [ 0,       0,     -0.008,   0.003, 0.005],
    [ 0,       0,      0,      -0.009, 0.009],
    [ 0,       0,      0,       0,     0    ]
])

def simulate_lifetimes(Q, n_patients=1000):
    lifetimes = np.zeros(n_patients)
    for i in range(n_patients):
        state = 0
        current_time = 0.0
        while state != 4:  # State 4 is Death
            lambda_i = -Q[state, state]
            current_time += np.random.exponential(1 / lambda_i)
            
            probs = Q[state, :].copy()
            probs[state] = 0
            probs = probs / lambda_i
            state = np.random.choice(5, p=probs)
            
        lifetimes[i] = current_time
    return lifetimes

# Run simulations
n_women = 1000
lifetimes_control = simulate_lifetimes(Q_control, n_women)
lifetimes_treatment = simulate_lifetimes(Q_treatment, n_women)

def plot_kaplan_meier(lifetimes, label, color):
    # Sort times from earliest death to latest
    t_sorted = np.sort(lifetimes)
    # Calculate proportion alive: (N - d(t)) / N
    # Since we sorted the times, the proportion drops by 1/N at each death
    survival_prob = np.linspace(1, 0, len(t_sorted), endpoint=False)
    
    # Use a step plot for the standard Kaplan-Meier aesthetic
    plt.step(t_sorted, survival_prob, where='post', label=label, color=color, linewidth=2)

plt.figure(figsize=(10, 6))

plot_kaplan_meier(lifetimes_control, label="Control (No Treatment)", color="red")
plot_kaplan_meier(lifetimes_treatment, label="Treatment", color="blue")

plt.title("Kaplan-Meier Survival Curves: Treatment vs. Control")
plt.xlabel("Months After Surgery (t)")
plt.ylabel("Survival Proportion $S(t)$")
plt.xlim(0, max(max(lifetimes_control), max(lifetimes_treatment)))
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.4)
plt.show()