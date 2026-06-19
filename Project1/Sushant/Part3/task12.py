import numpy as np
import bisect

# Generator matrix (Baseline Control Q)
Q = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [ 0,     -0.014,  0.005,  0.004, 0.005],
    [ 0,      0,     -0.008,  0.003, 0.005],
    [ 0,      0,      0,     -0.009, 0.009],
    [ 0,      0,      0,      0,     0    ]
])

n_women = 1000
all_time_series = []

for i in range(n_women):
    # --- 1. Simulate Continuous Path ---
    state = 0
    current_time = 0.0
    
    time_history = [0.0]
    state_history = [0]
    
    while state != 4:
        lambda_i = -Q[state, state]
        dt = np.random.exponential(1 / lambda_i)
        current_time += dt
        
        probs = Q[state, :].copy()
        probs[state] = 0
        probs = probs / lambda_i
        
        state = np.random.choice(5, p=probs)
        
        time_history.append(current_time)
        state_history.append(state)
        
    # --- 2. Extract Discrete Time Series Y(i) ---
    t_obs = 0.0
    series = []
    
    while True:
        # bisect_right finds the time interval containing t_obs
        idx = bisect.bisect_right(time_history, t_obs) - 1
        observed_state = state_history[idx]
        
        # Convert 0-based Python index to 1-based clinical state (1 to 5)
        series.append(observed_state + 1)
        
        # Terminate series if Death (State 5) is observed
        if observed_state == 4: 
            break
            
        t_obs += 48.0
        
    all_time_series.append(series)

# --- Output Verification ---
print("First 10 simulated time series Y(i):\n")
for idx, series in enumerate(all_time_series[:10]):
    print(f"Y({idx + 1}): {series}")