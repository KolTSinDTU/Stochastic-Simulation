import random
import math
from scipy.stats import chisquare
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# GLOBAL PARAMETERS & DATA
# =============================================================================

# ELI5: The Transition Matrix (P) is like a "Probability Map". 
# Each row represents where you are NOW, and each column is where you might go NEXT.
# State 1: Healthy, State 2: Local Recurrence, State 3: Distant Metastasis, State 4: Both, State 5: Death
TRANSITION_MATRIX = [
    [0.9915, 0.0050, 0.0025, 0.0000, 0.0010], # From State 1
    [0.0000, 0.9860, 0.0050, 0.0040, 0.0050], # From State 2
    [0.0000, 0.0000, 0.9920, 0.0030, 0.0050], # From State 3
    [0.0000, 0.0000, 0.0000, 0.9910, 0.0090], # From State 4
    [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]  # From State 5 (Death is "absorbing", you stay there)
]

STATE_LABELS = ["Healthy", "Local", "Distant", "Both", "Death"]

# Hint for NumPy: You can turn this into a 'NumPy array' to do fast math:
# P = np.array(TRANSITION_MATRIX)
# Then you can do P @ P (matrix multiplication) or np.linalg.matrix_power(P, 120).

# =============================================================================
# REUSABLE SIMULATION ENGINE
# =============================================================================

def run_single_simulation(start_state=1, max_months=None):
    """
    ELI5: This function simulates ONE woman's journey. 
    It 'rolls the dice' every month to see if she stays in her current state 
    or moves to a new one based on the probabilities in the matrix.
    """
    current_state = start_state
    months = 0
    history = [current_state] # We can keep track of every state she visited
    
    # We keep going until she reaches State 5 (Death)
    # If max_months is set (for Task 2), we might stop early.
    while current_state != 5:
        if max_months is not None and months >= max_months:
            break
            
        # Get the row of probabilities for our current state
        # Note: Indexing starts at 0, so State 1 is index 0.
        probs = TRANSITION_MATRIX[current_state - 1]
        
        # --- HOW TO PICK THE NEXT STATE ---
        # Plain Python Hint: 
        # 1. Generate a random number between 0 and 1: r = random.random()
        # 2. Check which "bucket" it falls into using the probabilities.
        # r = random.random()
        # cumulative_prob = 0.0
        # next_state = None
        # for i, p in enumerate(probs):
        #     cumulative_prob += p
        #     if r < cumulative_prob:
        #         next_state = i + 1 # +1 because states are 1-indexed
        #         break
        
        # ELI5: Imagine a 1-meter ruler. We divide it into sections based on probabilities.
        # If the probability to stay is 0.9915, the first section is 99.15cm long.
        # We throw a dart (random number); where it lands tells us the next state!
        
        # Placeholder for your "dart throwing" logic:
        # next_state = ... 
        next_state = np.random.choice([1, 2, 3, 4, 5], p=probs) 
        current_state = next_state
        
        # NumPy Hint:
        # next_state = np.random.choice([1, 2, 3, 4, 5], p=probs)
        
        # For now, let's just pretend we stay (replace this with your logic!)
        # current_state = next_state 
        
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

# =============================================================================
# TASK 1: BASIC SIMULATION
# =============================================================================

def solve_task_1():
    print("\n--- Task 1: Starting Simulation of 1000 Women ---")
    
    # ELI5: We want to see how long women live after surgery on average 
    # and how many of them see the cancer come back locally.
    
    lifetimes = []
    local_recurrence_count = 0
    num_simulations = 1000
    
    for _ in range(num_simulations):
        result = run_single_simulation(start_state=1)
        # Store the data you need...
        lifetimes.append(result["lifetime"])
        if result["had_local_recurrence"]:
            local_recurrence_count += 1

    # # --- VISUALIZATION HINTS ---
    # # To make a histogram (a bar chart showing how many lived 10 months, 20 months, etc.):
    # plt.hist(lifetimes, bins=30, color='skyblue', edgecolor='black')
    # # To add labels and title:
    # plt.title("Lifetime Distribution After Surgery")
    # plt.xlabel("Lifetime (Months)")
    # plt.ylabel("Number of Women")
    # plt.title("Lifetime Distribution")
    # plt.show()
    
    # # To calculate proportion:
    # # proportion = count / total
    # proportion = local_recurrence_count / num_simulations
    # print(f"Task 1 placeholders ready. Complete the loop and calculations!")
    # print(f"Proportion of women with local recurrence: {proportion:.2f}")

    return lifetimes # Return this for Task 3!

# =============================================================================
# TASK 2: STATE DISTRIBUTION AT T=120
# =============================================================================

def solve_task_2():
    print("\n--- Task 2: State Distribution at Month 120 ---")
    
    # ELI5: If we look at 1000 women exactly 10 years (120 months) later,
    # how many are Healthy, how many are in treatment, and how many have passed away?
    num_simulations = 1000
    final_states = []
    
    # Step 1: Run simulations but stop them at month 120.
    for _ in range(num_simulations):
        result = run_single_simulation(start_state=1, max_months=120)
        final_states.append(result["final_state"])

    # Step 2: Count how many women are in each state (1, 2, 3, 4, 5).
    state_counts = {state: 0 for state in range(1, 6)}
    for state in final_states:
        state_counts[state] += 1
     
    print("State distribution at month 120 (Simulation):")
    observed_counts = [state_counts[state] for state in range(1, 6)]
     
    print("\nState distribution at month 120 (Simulation):")
    for state, count in state_counts.items():
        # USING YOUR LABELS HERE:
        label = STATE_LABELS[state - 1] 
        print(f"{label} (State {state}): {count} women")

    # Step 3: Compare with Math (Theoretical)
    # Define P and the initial state
    P = np.array(TRANSITION_MATRIX)
    p_0 = np.array([1, 0, 0, 0, 0])
    
    # Calculate exact mathematical probabilities at month 120
    expected_probs = p_0 @ np.linalg.matrix_power(P, 120)
    
    # Convert probabilities to expected numbers of women (out of 1000)
    expected_counts = expected_probs * num_simulations

    print("\nState distribution at month 120 (Theoretical Expectation):")
    for i in range(5):
        print(f"{STATE_LABELS[i]}: {expected_counts[i]:.2f} women")

    # Step 4: The Chi-Square Goodness of Fit Test
    print("\n--- Statistical Test ---")
    chi_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    print(f"Chi-Square Statistic: {chi_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("Conclusion: The p-value is > 0.05. The simulated distribution corresponds to our expectations!")
    else:
        print("Conclusion: The p-value is < 0.05. The simulated distribution differs significantly from expectations.")
    
    print("Task 2 completed.")

# =============================================================================
# TASK 3: LIFETIME DISTRIBUTION VALIDATION
# =============================================================================


def solve_task_3(simulated_lifetimes):
    print("\n--- Task 3: Comparing Simulation to Phase-Type Theory ---")
    
    # 1. Setup the matrix
    P = np.array(TRANSITION_MATRIX)  
    
    # pi: Initial distribution for states 1-4. 100% start in State 1.
    pi = np.array([1, 0, 0, 0]) 
    
    # P_s: The 4x4 sub-matrix (Row 0-3, Col 0-3)
    P_s = P[:4, :4]
    
    # p_s: The probability of dying from states 1-4 (Row 0-3, Col 4)
    p_s = P[:4, 4] 
    
    # 2. Calculate Expected Lifetime E(T) as a sanity check!
    I = np.eye(4) # Creates a 4x4 identity matrix
    inverse_matrix = np.linalg.inv(I - P_s)
    expected_lifetime = pi @ inverse_matrix @ np.ones(4)
    
    print(f"Theoretical Mean Lifetime: {expected_lifetime:.2f} months")
    print(f"Simulated Mean Lifetime: {np.mean(simulated_lifetimes):.2f} months")
    
    # 3. Calculate the exact theoretical probability for every month t
    max_months = max(simulated_lifetimes)
    t_values = np.arange(max_months)
    theoretical_probs = []
    
    for t in t_values:
        # P_s^t (Raising the matrix to the power of t)
        P_s_to_t = np.linalg.matrix_power(P_s, t)
        
        # P(T=t) = pi * (P_s)^t * p_s
        prob_t = pi @ P_s_to_t @ p_s
        theoretical_probs.append(prob_t)
        
    # 4. Visual Comparison
    # IMPORTANT: Notice density=True! This changes your histogram from 
    # "Count of Women" to "Probability", so it matches the math scale.
    plt.hist(simulated_lifetimes, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Simulated (Histogram)')
    
    # Overlay the theoretical math formula as a red line
    plt.plot(t_values, theoretical_probs, color='red', linewidth=2, label='Theoretical (Formula)')
    
    plt.xlabel("Lifetime (Months)")
    plt.ylabel("Probability")
    plt.title("Task 3: Simulated vs. Theoretical Lifetime")
    plt.legend()
    plt.show()
    
    print("Task 3 completed. If the red line hugs the blue bars, your simulation is perfect!")

# =============================================================================
# TASK 4: REJECTION SAMPLING
# =============================================================================

def solve_task_4():
    print("\n--- Task 4: Rejection Sampling for Subgroups ---")
    
    # ELI5: We only want to study women who survived the first year BUT had a 
    # recurrence in that same year. Most women won't fit this!
    # "Rejection Sampling" means: keep simulating women, and if they don't 
    # fit our criteria, just 'reject' them and try again until we have 1000 who DO fit.
    
    accepted_simulations = []
    total_attempts = 0
    
    while len(accepted_simulations) < 1000:
        total_attempts += 1
        result = run_single_simulation(start_state=1)
        
        # Check criteria:
        # 1. Did she survive at least 12 months? (lifetime > 12)
        # 2. Did state 2, 3, or 4 appear in her history within the first 12 months?
        
        # Hint: result["history"][:13] gives you the states in the first 12 months.
        # if ... :
        #     accepted_simulations.append(result)
        pass

    print(f"Task 4: Found 1000 matches after {total_attempts} attempts.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # You can uncomment these as you work on them!
    #solve_task_1()
    # solve_task_2()
     my_lifetimes = solve_task_1()

     solve_task_3(simulated_lifetimes=my_lifetimes) # Replace with your actual simulated lifetimes
    # solve_task_4()
