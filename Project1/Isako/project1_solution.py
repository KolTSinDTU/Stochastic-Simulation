import random
import math
from scipy.stats import chisquare
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest
from scipy.linalg import expm

TRANSITION_MATRIX = [
    [0.9915, 0.0050, 0.0025, 0.0000, 0.0010], # From State 1
    [0.0000, 0.9860, 0.0050, 0.0040, 0.0050], # From State 2
    [0.0000, 0.0000, 0.9920, 0.0030, 0.0050], # From State 3
    [0.0000, 0.0000, 0.0000, 0.9910, 0.0090], # From State 4
    [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]  # From State 5 (Death is "absorbing", you stay there)
]

STATE_LABELS = ["Healthy", "Local", "Distant", "Both", "Death"]

# Define the Q-Matrix (Transition-Rate Matrix) for the continuous-time Markov process
Q = np.array([
    [-0.0085, 0.005,  0.0025, 0,      0.001],
    [ 0,     -0.014,  0.005,  0.004,  0.005],
    [ 0,      0,     -0.008,  0.003,  0.005],
    [ 0,      0,      0,     -0.009,  0.009],
    [ 0,      0,      0,      0,      0    ] # Death state has rates of 0
])

Q_treated = np.array([
    [-0.00475, 0.0025, 0.00125, 0,     0.001], # Recurrence rates halved
    [ 0,      -0.0095, 0.0025,  0.002, 0.005], # Recurrence rates halved
    [ 0,       0,     -0.008,   0.003, 0.005], # Same as before
    [ 0,       0,      0,      -0.009, 0.009], # Same as before
    [ 0,       0,      0,       0,     0    ]
])

def run_single_simulation(start_state=1, max_months=None):
    """
    ELI5: This function simulates ONE woman's journey. 
    stays in her current state 
    or moves to a new one based on the probabilities in the matrix.
    """
    current_state = start_state
    months = 0
    history = [current_state] # We can keep track of every state she visited
    
    # loop till death state 5 or limit
    while current_state != 5:
        if max_months is not None and months >= max_months:
            break
            
        probs = TRANSITION_MATRIX[current_state - 1]
        
        #get random 
        next_state = np.random.choice([1, 2, 3, 4, 5], p=probs) 
        current_state = next_state
        
        months += 1
        history.append(current_state)
        
        # Safety break to avoid infinite loops while you are coding:
        if months > 10000: 
            print("Warning: Hit 10,000 months. Exiting loop.")
            break
        
    return {
        "lifetime": months,
        "history": history,
        "final_state": current_state,
        "had_local_recurrence": 2 in history or 4 in history # Did she ever enter state 2 or 4?
    }

def run_ctmc_simulation(start_state=1):
    current_state = start_state
    current_time = 0.0
    
    # track distant recurrence happens
    time_distant_recurrence = None 
    
    while current_state != 5:
        # get index for numpy
        row_idx = current_state - 1
        
        # The total rate of leaving the current state is the negative diagonal element
        rate_out = -Q[row_idx, row_idx]
        
        # fet sojourn time: exponential distribution with mean 1/rate_out
        sojourn_time = np.random.exponential(scale=1.0 / rate_out)
        current_time += sojourn_time
        
        # get next state: probabilities are the off-diagonal elements divided by rate_out
        rates = Q[row_idx, :].copy()
        rates[row_idx] = 0 
        probs = rates / rate_out
        
        next_state = np.random.choice([1, 2, 3, 4, 5], p=probs)
        
        # Check distant recurrence (entering State 3 or 4)
        if next_state in [3, 4] and time_distant_recurrence is None:
            time_distant_recurrence = current_time
            
        current_state = next_state
        
    return {
        "lifetime": current_time,
        "time_distant_recurrence": time_distant_recurrence
    }

def run_treated_simulation(start_state=1):
    """A copy of our CTMC simulation that specifically uses the Treated Matrix."""
    current_state = start_state
    current_time = 0.0
    
    while current_state != 5:
        row_idx = current_state - 1
        rate_out = -Q_treated[row_idx, row_idx]
        
        # Time jump
        sojourn_time = np.random.exponential(scale=1.0 / rate_out)
        current_time += sojourn_time
        
        # State jump
        rates = Q_treated[row_idx, :].copy()
        rates[row_idx] = 0 
        probs = rates / rate_out
        
        current_state = np.random.choice([1, 2, 3, 4, 5], p=probs)
        
    return current_time

def simulate_bridge(start_state, end_state, Q_curr):
    """
    Simulates exactly 48 months. If it lands on 'end_state', we keep it!
    If it lands somewhere else, we throw it away and try again.
    """
    while True:
        curr = start_state
        t = 0.0
        
        # Track jumps and time spent in states for THIS 48-month window
        jumps = np.zeros((5, 5))
        sojourn = np.zeros(5)
        
        while t < 48.0 and curr != 5:
            rate_out = -Q_curr[curr - 1, curr - 1] #q_curr is the current guess for the Q matrix, so we use that to get the rate out of the current state. This is the negative of the diagonal element for that state.
            
            # Draw time until next jump
            dt = np.random.exponential(1.0 / rate_out)
            
            if t + dt >= 48.0: # 48 months is up before the next jump
                sojourn[curr - 1] += (48.0 - t)
                t = 48.0
                break
            else:
                #jump before 48 months
                sojourn[curr - 1] += dt
                t += dt
                
               #get next state
                rates = Q_curr[curr - 1, :].copy()
                rates[curr - 1] = 0
                probs = rates / rate_out
                
                next_state = np.random.choice([1, 2, 3, 4, 5], p=probs)
                jumps[curr - 1, next_state - 1] += 1
                curr = next_state
                
        # stay dead if dead at 48 months
        if curr == 5 and t < 48.0:
            sojourn[4] += (48.0 - t)
            
        # 
        if curr == end_state:
            return jumps, sojourn


def solve_task_13(observations):
        
    # Initial Guess
    #Q_curr = Q * 1.5
    Q_curr = np.array([ [-0.01275, 0.0075, 0.00375, 0,      0.0015],
                        [ 0,      -0.021,  0.0075, 0.006,  0.0075],
                        [ 0,       0,     -0.012,  0.0045, 0.0075],
                        [ 0,       0,      0,     -0.0135, 0.0135],
                        [ 0,       0,      0,      0,      0    ]])
    
    iteration = 0
    max_iterations = 20 # Safety break
    
    while iteration < max_iterations:
        iteration += 1
        
        N_total = np.zeros((5, 5))
        S_total = np.zeros(5)
        
        for i, obs in enumerate(observations):
            # Print progress so you know it hasn't crashed
            if (i + 1) % 200 == 0:
                print(f"  ... bridged gaps for {i + 1} women ...")
                
            for m in range(len(obs) - 1):
                start_s = obs[m]
                end_s = obs[m + 1]
                
                # Get the accepted simulated history between these two checkups
                jumps, sojourn = simulate_bridge(start_s, end_s, Q_curr)
                N_total += jumps
                S_total += sojourn
                
        # STEP 2: The M-Step (Calculate the new Q matrix)
        Q_new = np.zeros((5, 5))
        for i in range(4): # We don't calculate rates OUT of state 5
            for j in range(5):
                if i != j and S_total[i] > 0:
                    Q_new[i, j] = N_total[i, j] / S_total[i]
                    
            # The diagonal must be the negative sum of the rest of the row
            Q_new[i, i] = -np.sum(Q_new[i, :])
            
        # STEP 3: Check for Convergence
        # Find the single biggest change between the old matrix and the new matrix
        max_diff = np.max(np.abs(Q_new - Q_curr))
        print(f"-> Iteration {iteration} Complete. Max Difference: {max_diff:.5f}")
        
        Q_curr = Q_new.copy()
        
        if max_diff < 1e-3:
            print("\nConvergence criterion reached! The matrix has stabilized.")
            break
            
    print("\n--- Final Estimated Q Matrix ---")
    print(np.round(Q_curr, 5))
    print("\n--- True Q Matrix (For Comparison) ---")
    print(np.round(Q, 5))



def solve_task_1():
    #Summarize the lifetime distribution of the
    #women, after surgery, for example using a histogram. In what proportion of
    #women does the cancer eventually reappear, locally?
    
    lifetimes = []
    local_recurrence_count = 0
    num_simulations = 1000
    
    for _ in range(num_simulations):
        result = run_single_simulation(start_state=1)
        # Store the data you need...
        lifetimes.append(result["lifetime"])
        if result["had_local_recurrence"]:
            local_recurrence_count += 1

    # --- VISUALIZATION HINTS ---
    # To make a histogram (a bar chart showing how many lived 10 months, 20 months, etc.):
    plt.hist(lifetimes, bins=30, color='skyblue', edgecolor='black')
    # To add labels and title:
    plt.title("Lifetime Distribution After Surgery")
    plt.xlabel("Lifetime (Months)")
    plt.ylabel("Number of Women")
    plt.title("Lifetime Distribution")
    plt.show()
    
    #calculate the proportion
    proportion = local_recurrence_count / num_simulations
    print(f"Task 1 placeholders ready. Complete the loop and calculations!")
    print(f"Proportion of women with local recurrence: {proportion:.2f}")
    plt.show()

    return lifetimes # Return this for Task 3!

def solve_task_2():
    #In your simulations, what is the distribution over the states at t = 120?
    #Does this correspond to what we expect?
    num_simulations = 1000
    final_states = []
    
    #stop them at month 120.
    for _ in range(num_simulations):
        result = run_single_simulation(start_state=1, max_months=120)
        final_states.append(result["final_state"])

    #Count how many women are in each state (1, 2, 3, 4, 5).
    state_counts = {state: 0 for state in range(1, 6)}
    for state in final_states:
        state_counts[state] += 1
    
    print("State distribution at month 120 (Simulation):")
    observed_counts = [state_counts[state] for state in range(1, 6)]
     
    print("\nState distribution at month 120 (Simulation):")
    for state, count in state_counts.items():
        label = STATE_LABELS[state - 1] 
        print(f"{label} (State {state}): {count} women")

    P = np.array(TRANSITION_MATRIX)
    p_0 = np.array([1, 0, 0, 0, 0])
    
    # Calculate prob at month 120
    expected_probs = p_0 @ np.linalg.matrix_power(P, 120)
    
    # get women counts prob * simulations
    expected_counts = expected_probs * num_simulations

    print("\nState distribution at month 120 (Theoretical Expectation):")
    for i in range(5):
        print(f"{STATE_LABELS[i]}: {expected_counts[i]:.2f} women")

    #The Chi-Square Goodness of Fit Test
    print("\n--- Statistical Test ---")
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    print(f"Chi-Square Statistic: {chi_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
def solve_task_3(simulated_lifetimes):
    # distribution of lifetimes compared to theoretical
    
    P = np.array(TRANSITION_MATRIX)  
    # pi: Initial distribution for states 1-4. 100% start in State 1.
    pi = np.array([1, 0, 0, 0]) 
    # P_s: The 4x4 sub-matrix (Row 0-3, Col 0-3)
    P_s = P[:4, :4]
    # p_s: The probability of dying from states 1-4 (Row 0-3, Col 4)
    p_s = P[:4, 4] 
    
    # Calculate Expected Lifetime E(T)!
    I = np.eye(4) # Creates a 4x4 identity matrix
    inverse_matrix = np.linalg.inv(I - P_s)
    expected_lifetime = pi @ inverse_matrix @ np.ones(4)
    
    print(f"Theoretical Mean Lifetime: {expected_lifetime:.2f} months")
    print(f"Simulated Mean Lifetime: {np.mean(simulated_lifetimes):.2f} months")
    
    # Calculate the exact theoretical probability for every month t
    max_months = max(simulated_lifetimes)
    t_values = np.arange(max_months)
    theoretical_probs = []
    
    for t in t_values:
        P_s_to_t = np.linalg.matrix_power(P_s, t)
        prob_t = pi @ P_s_to_t @ p_s
        theoretical_probs.append(prob_t)
        
    plt.hist(simulated_lifetimes, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Simulated (Histogram)')
    plt.plot(t_values, theoretical_probs, color='red', linewidth=2, label='Theoretical (Formula)')
    
    plt.xlabel("Lifetime (Months)")
    plt.ylabel("Probability")
    plt.title("Task 3: Simulated vs. Theoretical Lifetime")
    plt.legend()
    plt.show()

def solve_task_4():
    # Rejection Sampling" means: keep simulating women, and if they don't 
    # fit our criteria, just 'reject' them and try again until we have 1000 who DO fit.
    accepted_simulations = []
    total_attempts = 0
    
    while len(accepted_simulations) < 1000:
        total_attempts += 1
        result = run_single_simulation(start_state=1)
        
        # Check criteria: survive 12 months AND have a local or distant recurrence within those 12 months
        if result["lifetime"] > 12 and (2 in result["history"][:13] or 3 in result["history"][:13] or 4 in result["history"][:13]):
            accepted_simulations.append(result["lifetime"])
        
    expected_lifetime = np.mean(accepted_simulations)    
    print(f"Estimated Expected Lifetime for this subgroup: {expected_lifetime:.2f} months")
    print(f"Task 4: Found 1000 matches after {total_attempts} attempts.")

def solve_task_5():
    # Task 5: Variance Reduction using Control Variates
    # calculate true expected lifetime using the formula E(T) = pi * (I - P_s)^(-1) * 1
    P = np.array(TRANSITION_MATRIX)
    P_s = P[:4, :4]
    pi = np.array([1, 0, 0, 0])
    I = np.eye(4)
    expected_lifetime_true = pi @ np.linalg.inv(I - P_s) @ np.ones(4)

    Y_crude = []  # The fraction of women who die <= 350 months (The thing we want)
    X_means = []  # The mean lifetime of the 200 women (The Control Variate)
    
    num_batches = 100
    sims_per_batch = 200
    
    print(f"Simulating {num_batches} batches of {sims_per_batch} women")
    
    for _ in range(num_batches):
        batch_lifetimes = []
        
        for _ in range(sims_per_batch):
            result = run_single_simulation(start_state=1)
            batch_lifetimes.append(result["lifetime"])
            
        #fraction of women who died within 350 months
        died_within_350 = sum(1 for life in batch_lifetimes if life <= 350)
        fraction = died_within_350 / sims_per_batch
        Y_crude.append(fraction)
        
        # Calculate X: Mean lifetime of these 200 women
        mean_life = np.mean(batch_lifetimes)
        X_means.append(mean_life)
        
    Y_crude = np.array(Y_crude)
    X_means = np.array(X_means)
    
    # 3. Calculate the optimal adjustment factor (c*) using the covariance and variance of X and Y
    cov_matrix = np.cov(X_means, Y_crude)
    covariance_xy = cov_matrix[0, 1]
    variance_x = cov_matrix[0, 0]
    
    c_star = covariance_xy / variance_x
    
    # Formula: Y_adjusted = Y_crude - c * (X_mean - True_X_Mean)
    Y_cv = Y_crude - c_star * (X_means - expected_lifetime_true)
    
    #compare the variances
    var_crude = np.var(Y_crude, ddof=1)
    var_cv = np.var(Y_cv, ddof=1)
    
    #Variance reduction percentage:
    variance_reduction = (1 - (var_cv / var_crude)) * 100
    
    print("\n--- Results ---")
    print(f"Crude Estimate (Fraction dying <= 350 months): {np.mean(Y_crude):.4f}")
    print(f"CV Adjusted Estimate:                        {np.mean(Y_cv):.4f}")
    print("-" * 40)
    print(f"Crude Variance: {var_crude:.6f}")
    print(f"CV Variance:    {var_cv:.6f}")
    print(f"Variance Reduction: {variance_reduction:.2f}%")

def solve_task_7():
    #continous time simulations
    num_sims = 1000
    lifetimes_a = []
    distant_after_30_5_count = 0
    
    for _ in range(num_sims):
        result = run_ctmc_simulation(start_state=1)
        lifetimes_a.append(result["lifetime"])
        
        # check recurence after 30.5 months
        t_distant = result["time_distant_recurrence"]
        if t_distant is not None and t_distant > 30.5:
            distant_after_30_5_count += 1

    plt.hist(lifetimes_a, bins=30, color='lightgreen', edgecolor='black')
    plt.xlabel("Lifetime (Months)")
    plt.ylabel("Number of Women")
    plt.title("Continuous-Time Lifetime Distribution")
    plt.show()
    
    #Mean and Confidence Interval ---
    lifetimes = np.array(lifetimes_a)
    mean_lt = np.mean(lifetimes)
    # Standard Error of the Mean = std / sqrt(n)
    sem_lt = stats.sem(lifetimes) 
    # 95% CI for the Mean (using t-distribution)
    ci_mean = stats.t.interval(0.95, df=num_sims-1, loc=mean_lt, scale=sem_lt)
    
    # Standard Deviation and Confidence Interval ---
    std_lt = np.std(lifetimes, ddof=1)
    variance = np.var(lifetimes, ddof=1)
    
    # 95% CI for Variance (using Chi-Square distribution)
    chi2_lower = stats.chi2.ppf(0.025, df=num_sims-1)
    chi2_upper = stats.chi2.ppf(0.975, df=num_sims-1)
    ci_var = [(num_sims - 1) * variance / chi2_upper, (num_sims - 1) * variance / chi2_lower]
    
    # CI for Standard Deviation is just the square root of the variance CI
    ci_std = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))
    
    # --- 4. Distant Recurrence Proportion ---
    prop_distant = distant_after_30_5_count / num_sims
    
    print(f"Mean Lifetime: {mean_lt:.2f} months")
    print(f"95% CI for Mean: ({ci_mean[0]:.2f}, {ci_mean[1]:.2f})\n")
    
    print(f"Standard Deviation: {std_lt:.2f} months")
    print(f"95% CI for Std Dev: ({ci_std[0]:.2f}, {ci_std[1]:.2f})\n")
    
    print(f"Proportion with distant recurrence after 30.5 months: {prop_distant:.4f}")
    return lifetimes_a # for task 8

def solve_task_8(simulated_lifetimes):
    print("\n--- Task 8: Comparing Continuous Simulation to Theory ---")
    
    # 1. Define the continuous math components [cite: 121]
    Q = np.array([
        [-0.0085, 0.005,  0.0025, 0,      0.001],
        [ 0,     -0.014,  0.005,  0.004,  0.005],
        [ 0,      0,     -0.008,  0.003,  0.005],
        [ 0,      0,      0,     -0.009,  0.009],
        [ 0,      0,      0,      0,      0    ]
    ])
    
    # Q_s is the sub-matrix (removing the death state) [cite: 120]
    Q_s = Q[:4, :4]
    
    # p_0 is the initial state distribution (100% in state 1)
    p_0 = np.array([1, 0, 0, 0])
    ones = np.ones(4)
    
    # 2. Create the theoretical CDF function
    # kstest requires a function that takes an array of times and returns probabilities
    def theoretical_cdf(t_array):
        # Ensure it works even if kstest passes a single number instead of a list
        t_array = np.atleast_1d(t_array)
        cdf_values = []
        
        for t in t_array:
            # Formula: F(t) = 1 - p0 * exp(Qs * t) * 1
            matrix_exp = expm(Q_s * t) # This computes the matrix exponential [cite: 123, 125]
            prob_alive = p_0 @ matrix_exp @ ones
            cdf_values.append(1.0 - prob_alive)
            
        return np.array(cdf_values)
        
    # 3. Perform the Kolmogorov-Smirnov (K-S) Test
    # This compares our simulated lifetimes against the perfect math curve
    stat, p_value = kstest(simulated_lifetimes, theoretical_cdf)
    
    print("Kolmogorov-Smirnov Test Results:")
    print(f"Test Statistic (Max Gap between curves): {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("Conclusion: p-value > 0.05. We fail to reject the null hypothesis.")
        print("Your continuous simulation flawlessly matches the phase-type distribution theory!")
    else:
        print("Conclusion: p-value < 0.05. The simulated distribution differs significantly from theory.")
        
    # --- 4. VISUALIZATION ---
    # Seeing the curves overlap is incredibly satisfying!
    
    # Sort lifetimes to plot the Empirical CDF (staircase curve)
    x_sim = np.sort(simulated_lifetimes)
    y_sim = np.arange(1, len(x_sim) + 1) / len(x_sim)
    
    # Generate smooth points for the Theoretical CDF
    x_math = np.linspace(0, max(x_sim), 100)
    y_math = theoretical_cdf(x_math)
    
    plt.step(x_sim, y_sim, label='Simulated (ECDF)', color='blue')
    plt.plot(x_math, y_math, label='Theoretical (CDF)', color='red', linestyle='--')
    plt.xlabel('Lifetime (Months)')
    plt.ylabel('Cumulative Probability of Death')
    plt.title('Task 8: Simulated vs Theoretical Distribution')
    plt.legend()
    plt.show()

def solve_task_9(untreated_lifetimes):
    print("\n--- Task 9: Kaplan-Meier Survival Curves ---")
    
    # 1. Simulate 1000 women with the NEW treatment
    num_sims = 1000
    treated_lifetimes = []
    
    print("Simulating 1000 treated women...")
    for _ in range(num_sims):
        lifetime = run_treated_simulation(start_state=1)
        treated_lifetimes.append(lifetime)
        
    # 2. Calculate the Kaplan-Meier Data (The "Staircase")
    # Formula: S(t) = (N - deaths) / N
    
    def calculate_km_curve(lifetimes):
        # Sort from shortest life to longest life
        x_times = np.sort(lifetimes)
        # Calculate the percentage surviving at each step
        N = len(x_times)
        y_survival = 1.0 - (np.arange(1, N + 1) / N)
        
        # Add a starting point at time 0 where survival is 1.0 (100%)
        x_times = np.insert(x_times, 0, 0)
        y_survival = np.insert(y_survival, 0, 1)
        
        return x_times, y_survival

    x_untreated, y_untreated = calculate_km_curve(untreated_lifetimes)
    x_treated, y_treated = calculate_km_curve(treated_lifetimes)

    # 3. Plot both curves together
    # 'post' ensures the line draws out flat, then drops straight down like real stairs
    plt.step(x_untreated, y_untreated, where='post', label='Untreated (Task 7)', color='red')
    plt.step(x_treated, y_treated, where='post', label='Treated', color='green')
    
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability S(t)')
    plt.title('Kaplan-Meier Survival Curves: Treated vs Untreated')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Task 9 Complete! Look at the graph to answer the final question.")
    print("If the Treated (Green) curve is consistently HIGHER than the Untreated (Red) curve,")
    print("it means the treatment is effective at extending life.")

def solve_task_12():
    
    # Calculate the exact probability matrix for a 48-month jump
    P_48 = expm(Q * 48)
    
    #zero out tiny negatives
    P_48 = np.clip(P_48, 0, 1)
    
    # Re-normalize rows to ensure they sum to exactly 1.0
    for i in range(5):
        P_48[i] = P_48[i] / np.sum(P_48[i])
        
    observations = []
    
    for _ in range(1000):
        current_state = 1
        history = [current_state]
        
        while current_state != 5:
            probs = P_48[current_state - 1]
            current_state = np.random.choice([1, 2, 3, 4, 5], p=probs)
            history.append(current_state)
            
        observations.append(history)
        
    print("Successfully generated 48-month interval data for 1000 women!")
    print(f"Example Patient 1 History: {observations[0]}")
    print(f"Example Patient 2 History: {observations[1]}")

    #plot the distribution of final states at the end of 48 months
    # 1. Get the states at exactly the 48-month mark (which is index 1 in the history lists)
    states_at_48 = [history[1] for history in observations]
    state_labels = [1, 2, 3, 4, 5]
    counts_at_48 = [states_at_48.count(s) for s in state_labels]

    # 2. Prep data for the full timeline progression
    max_jumps = max(len(h) for h in observations)
    state_matrix = np.zeros((5, max_jumps))

    for h in observations:
        # Tally the actual recorded states
        for jump_idx, state in enumerate(h):
            state_matrix[state - 1, jump_idx] += 1
        # Once in state 5 (absorbing), carry that forward to the end of the matrix
        for remaining_idx in range(len(h), max_jumps):
            state_matrix[4, remaining_idx] += 1 

    # Convert counts to proportions (0.0 to 1.0)
    state_props = state_matrix / 1000.0
    time_axis = np.arange(max_jumps) * 48

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c', '#333333']

    # Plot A: Bar Chart for Month 48
    bars = ax1.bar(state_labels, counts_at_48, color=colors, alpha=0.8)
    ax1.set_title("Distribution of States at 48 Months")
    ax1.set_xlabel("Health State")
    ax1.set_ylabel("Number of Women (out of 1000)")
    ax1.set_xticks(state_labels)
    
    # Add count labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom')

    # Plot B: Stacked Area Chart for the whole timeline
    ax2.stackplot(time_axis, state_props, labels=['State 1', 'State 2', 'State 3', 'State 4', 'State 5 (Dead)'], 
                  colors=colors, alpha=0.8)
    ax2.set_title("Cohort Progression Over Time")
    ax2.set_xlabel("Months")
    ax2.set_ylabel("Proportion of Cohort")
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, time_axis[-1])
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    
    return observations

if __name__ == "__main__":
     solve_task_13(observations=solve_task_12())
    