import numpy as np
import bisect
import matplotlib.pyplot as plt

Q = np.array(
    [
        [-0.0085, 0.005, 0.0025, 0, 0.001],
        [0, -0.014, 0.005, 0.004, 0.005],
        [0, 0, -0.008, 0.003, 0.005],
        [0, 0, 0, -0.009, 0.009],
        [0, 0, 0, 0, 0],
    ]
)


def get_all_time_series():

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
    return all_time_series


def simulate_interval(start_state, end_state, Q, interval=48.0):
    while True:
        state = start_state
        t = 0.0

        # Track transitions and sojourn times for this specific attempt
        path_N = np.zeros((5, 5))
        path_S = np.zeros(5)

        while t < interval:
            lambda_i = -Q[state, state]

            # If in the absorbing state (Death), stay there for the remaining time
            if lambda_i == 0:
                path_S[state] += interval - t
                state = 4
                break

            # Sample time to next transition
            dt = np.random.exponential(1 / lambda_i)

            if t + dt >= interval:
                # Transition happens after the observation window closes
                path_S[state] += interval - t
                break
            else:
                # Transition happens within the window
                path_S[state] += dt
                t += dt

                # Determine next state
                probs = Q[state, :].copy()
                probs[state] = 0
                probs /= lambda_i
                next_state = np.random.choice(5, p=probs)

                # Record the jump
                path_N[state, next_state] += 1
                state = next_state

        # Rejection criterion: Only accept if the path ends exactly where we observed it
        if state == end_state:
            return path_N, path_S


def mcem_estimation(time_series_data, Q_initial, tol=1e-3, max_iter=20):
    Q_k = Q_initial.copy()

    for iteration in range(max_iter):

        # Total counts for this iteration
        N_total = np.zeros((5, 5))
        S_total = np.zeros(5)

        for series in time_series_data:
            # Loop through each 48-month observation interval
            for m in range(len(series) - 1):

                start_state = series[m] - 1
                end_state = series[m + 1] - 1

                path_N, path_S = simulate_interval(start_state, end_state, Q_k)

                N_total += path_N
                S_total += path_S

        Q_next = np.zeros((5, 5))
        for i in range(4):
            for j in range(5):
                if i != j:
                    # q_ij = N_ij / S_i
                    Q_next[i, j] = N_total[i, j] / S_total[i] if S_total[i] > 0 else 0

            # Diagonal elements: Negative sum of off-diagonals
            Q_next[i, i] = -np.sum(Q_next[i, :])

        # Check convergence
        diff = np.max(np.abs(Q_k - Q_next))

        if diff < tol:
            print("Convergence reached!")
            return Q_next

        Q_k = Q_next

    print("Reached maximum iterations without strict convergence.")
    return Q_k


# Provide a uniform initial guess (avoid zeros to ensure all paths are initially possible)
Q_guess = np.array(
    [
        [-0.04, 0.01, 0.01, 0.01, 0.01],
        [0.01, -0.12, 0.01, 0.09, 0.01],
        [0.01, 0.08, -0.11, 0.01, 0.01],
        [0.02, 0.01, 0.01, -0.34, 0.3],
        # [0.00, -0.03, 0.01, 0.01, 0.01],
        # [0.000, 0.000, -0.02, 0.01, 0.01],
        # [0.000, 0.000, 0.000, -0.01, 0.01],
        [0.000, 0.000, 0.000, 0.000, 0.000],
    ]
)

estimated_Q = mcem_estimation(get_all_time_series(), Q_guess)

print("\nEstimated Q Matrix:")
print(np.round(estimated_Q, 4))

print("Max diff with the true Q Matrix:")
print(f"\n {np.max(np.abs(estimated_Q - Q))}")


# For time series distribution visualiozation in task 12 and 13

def simulate_ctmc(Q, initial_state, n_steps, exit_states=[]):
    states = [initial_state]
    times = [0]
    current_state = states[-1]
    i = 0
    while i != n_steps:
        if current_state in exit_states:
            break
        rate = -Q[current_state, current_state]
        if rate == 0:
            break
        time_to_next = np.random.exponential(1/rate)
        times.append(times[-1] + time_to_next)
        possible_next_states = np.where(Q[current_state] > 0)[0]
        probabilities = Q[current_state, possible_next_states] / rate
        probabilities = probabilities / np.sum(probabilities)  # normalize to ensure sum to 1
        next_state = np.random.choice(possible_next_states, p=probabilities)
        states.append(next_state)
        current_state = next_state
        i += 1
    return states, times

def visualize_ctmc_time_series(Q):
    ctmc_samples_estimated = []
    for _ in range(1000):
        print(f"Simulating sample {_+1}/1000", end="\r")
        ctmc_samples_estimated.append(simulate_ctmc(Q, initial_state=0, n_steps=-1, exit_states=[4]))

    ctmc_time_series_estimated = []
    for states, times in ctmc_samples_estimated:
        time_points = np.arange(0, times[-1]+1, 48)
        series = []
        for t in time_points:
            idx = np.searchsorted(times, t, side='right') - 1
            series.append(states[idx])
        ctmc_time_series_estimated.append(series)

    ctmc_state_counts = np.zeros((len(Q), max(len(s) for s in ctmc_time_series_estimated)))
    for series in ctmc_time_series_estimated:
        series = series + [4] * (ctmc_state_counts.shape[1] - len(series)) # pad series with dead state if shorter than max length
        for t, state in enumerate(series):
            ctmc_state_counts[state, t] += 1


    plt.figure(figsize=(8, 6))

    colors = ['#fde725', '#5ec962', '#21918c', '#3b528b', '#440154']
    state_labels = ['Healthy', 'Local Recurrence', 'Distant Recurrence', 'Both Recurrences', 'Death']

    for i in range(len(Q)):
        plt.plot(ctmc_state_counts[i, :], marker='o', label=state_labels[i], color=colors[i])
    plt.xticks(np.arange(0, ctmc_state_counts.shape[1],2), 4*np.arange(0, ctmc_state_counts.shape[1],2))
    plt.xlabel('Time (years)')
    plt.ylabel('Count')
    plt.title('State Counts Over Time with 48-month Intervals (Estimated Q)')
    plt.legend()
    plt.show()