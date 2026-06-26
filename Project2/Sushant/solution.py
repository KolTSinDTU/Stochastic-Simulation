import numpy as np
import heapq
import math
import random

SIM_DAYS = 365


def lambda_1(t):
    # Ward A arrival rate (Polynomial)
    return -(1 / 3650) * t**2 + (1 / 10) * t


def lambda_2(t):
    # Ward B arrival rate
    return 0.2 * lambda_1(t)


def lambda_3():
    # Ward C constant arrival rate
    return 6.0


def get_los_lognormal(ward_type):
    # Returns length-of-stay sampled from the log-normal distribution
    sigma = np.sqrt(np.log(2))
    if ward_type == "A":
        mu = np.log(4 * np.sqrt(2))
    elif ward_type == "B":
        mu = np.log(6 * np.sqrt(2))
    elif ward_type == "C":
        mu = np.log(5 * np.sqrt(2))
    return np.random.lognormal(mu, sigma)


def get_los_exponential(ward_type):
    # Alternative exponential distribution for sensitivity analysis
    if ward_type == "A":
        return np.random.exponential(8)
    elif ward_type == "B":
        return np.random.exponential(12)
    elif ward_type == "C":
        return np.random.exponential(10)


def generate_arrivals(lambda_func, total_days):
    # Generates arrivals using the Thinning (Acceptance-Rejection) algorithm
    arrivals = []
    
    # 1. Determine the global maximum rate (max_lambda) over the timeline.
    # Sampling 1000 points guarantees we catch the peak of the quadratic curve 
    # without needing to pass the exact derivative maximum manually.
    t_samples = np.linspace(0, total_days, 1000)
    max_lambda = max(lambda_func(t) for t in t_samples)
    
    # If the max rate is zero or negative across the board, no arrivals happen.
    if max_lambda <= 0:
        return arrivals

    t = 0
    while t < total_days:
        # 2. Generate candidate arrival time using a Homogeneous Poisson Process 
        # at the maximum possible rate.
        u1 = np.random.uniform(0, 1)
        t -= np.log(u1) / max_lambda
        
        if t > total_days:
            break
            
        # 3. Acceptance/Rejection phase
        # Evaluate the true arrival rate exactly at candidate time t
        current_rate = lambda_func(t)
        
        # We cannot have a negative rate in a Poisson process. Treat as 0.
        if current_rate < 0:
            current_rate = 0
            
        u2 = np.random.uniform(0, 1)
        # Accept the candidate arrival with probability proportional to the true rate
        if u2 <= (current_rate / max_lambda):
            arrivals.append(t)
            
    return arrivals

def generate_hpp_arrivals(rate, t_end):
    # Standard Homogeneous Poisson Process generation for static rates
    arrivals = []
    
    # Safeguard against zero/negative rates causing infinite loops or math errors
    if rate <= 0:
        return arrivals
        
    t = 0
    while t < t_end:
        u = np.random.uniform(0, 1)
        t -= np.log(u) / rate
        if t > t_end:
            break
        arrivals.append(t)
        
    return arrivals


def simulate_hospital(beds_A, beds_B, beds_C, use_exponential=False):
    # Generate all arrivals using the appropriate sampling method
    arr_A = generate_arrivals(lambda_1, SIM_DAYS)
    arr_B = generate_arrivals(lambda_2, SIM_DAYS)
    arr_C = generate_hpp_arrivals(lambda_3(), SIM_DAYS)

    # Event queue: (time, event_type, patient_type, assigned_ward)
    # event_type: 0 for departure, 1 for arrival (ensures departures process first if times match)
    events = []
    for t in arr_A:
        heapq.heappush(events, (t, 1, "A", None))
    for t in arr_B:
        heapq.heappush(events, (t, 1, "B", None))
    for t in arr_C:
        heapq.heappush(events, (t, 1, "C", None))

    # State variables
    occupied = {"A": 0, "B": 0, "C": 0}
    capacity = {"A": beds_A, "B": beds_B, "C": beds_C}

    # Statistics
    stats = {
        "arrivals": {"A": 0, "B": 0, "C": 0},
        "relocated": {"A": 0, "B": 0, "C": 0},
        "blocked_on_arrival": {"A": 0, "B": 0, "C": 0},
    }

    # For utilization (area under the curve of occupancy)
    last_time = 0
    occupancy_integral = {"A": 0.0, "B": 0.0, "C": 0.0}

    while events:
        t, e_type, p_type, assigned_ward = heapq.heappop(events)

        # Update integrals for utilization
        dt = t - last_time
        for w in ["A", "B", "C"]:
            occupancy_integral[w] += occupied[w] * dt
        last_time = t

        if e_type == 1:  # Arrival
            stats["arrivals"][p_type] += 1
            placed = False

            if p_type == "A":
                if occupied["A"] < capacity["A"]:
                    occupied["A"] += 1
                    assigned_ward = "A"
                    placed = True
                else:
                    stats["blocked_on_arrival"]["A"] += 1
                    stats["relocated"]["A"] += 1

            elif p_type == "B":
                if occupied["B"] < capacity["B"]:
                    occupied["B"] += 1
                    assigned_ward = "B"
                    placed = True
                else:
                    stats["blocked_on_arrival"]["B"] += 1
                    # Overflow to Ward A
                    if occupied["A"] < capacity["A"]:
                        occupied["A"] += 1
                        assigned_ward = "A"
                        placed = True
                    else:
                        stats["relocated"]["B"] += 1

            elif p_type == "C":
                if occupied["C"] < capacity["C"]:
                    occupied["C"] += 1
                    assigned_ward = "C"
                    placed = True
                else:
                    stats["blocked_on_arrival"]["C"] += 1
                    stats["relocated"]["C"] += 1

            if placed:
                los = (
                    get_los_exponential(p_type)
                    if use_exponential
                    else get_los_lognormal(p_type)
                )
                heapq.heappush(events, (t + los, 0, p_type, assigned_ward))

        else:  # Departure
            occupied[assigned_ward] -= 1

    # Add remaining time to end of simulation for utilization calculation
    dt = SIM_DAYS - last_time
    if dt > 0:
        for w in ["A", "B", "C"]:
            occupancy_integral[w] += occupied[w] * dt

    utilization = {
        w: occupancy_integral[w] / (capacity[w] * SIM_DAYS) if capacity[w] > 0 else 0
        for w in ["A", "B", "C"]
    }

    return stats, utilization


def get_expected_arrivals():
    # Calculate the exact mathematical expectation for the piecewise constant process
    expected_A = sum(max(0, lambda_1(d + 0.5)) for d in range(SIM_DAYS))
    expected_B = 0.2 * expected_A
    expected_C = lambda_3() * SIM_DAYS
    return expected_A + expected_B + expected_C


def run_multiple_simulations(
    beds_A, beds_B, beds_C, iterations=50, use_exponential=False
):
    # Runs the simulation multiple times and applies Control Variate for variance reduction
    results_Y = []  # Target variable: total relocated
    results_C = []  # Control variate: total arrivals

    tot_relocated = {"A": 0, "B": 0, "C": 0}
    tot_arrivals = {"A": 0, "B": 0, "C": 0}
    tot_blocked = {"A": 0, "B": 0, "C": 0}
    tot_util = {"A": 0.0, "B": 0.0, "C": 0.0}

    for _ in range(iterations):
        stats, util = simulate_hospital(beds_A, beds_B, beds_C, use_exponential)

        sim_relocated = sum(stats["relocated"].values())
        sim_arrivals = sum(stats["arrivals"].values())

        results_Y.append(sim_relocated)
        results_C.append(sim_arrivals)

        for w in ["A", "B", "C"]:
            tot_relocated[w] += stats["relocated"][w]
            tot_arrivals[w] += stats["arrivals"][w]
            tot_blocked[w] += stats["blocked_on_arrival"][w]
            tot_util[w] += util[w]

    # --- CONTROL VARIATE ADJUSTMENT ---
    mean_Y = np.mean(results_Y)

    # Fallback to crude mean if there is zero variance in arrivals (e.g., edge case seed)
    if np.var(results_C) == 0:
        adjusted_mean_relocated = mean_Y
    else:
        # Calculate optimal coefficient beta = Cov(Y, C) / Var(C)
        covariance = np.cov(results_Y, results_C)[0, 1]
        variance_C = np.var(results_C, ddof=1)
        beta = covariance / variance_C

        expected_C = get_expected_arrivals()
        mean_C = np.mean(results_C)

        # Adjust the crude estimator
        adjusted_mean_relocated = mean_Y - beta * (mean_C - expected_C)

    # Compile final statistics
    results = {
        "prob_full_on_arrival": {
            w: (tot_blocked[w] / tot_arrivals[w]) if tot_arrivals[w] > 0 else 0
            for w in ["A", "B", "C"]
        },
        "mean_relocated": {w: tot_relocated[w] / iterations for w in ["A", "B", "C"]},
        "mean_utilization": {w: tot_util[w] / iterations for w in ["A", "B", "C"]},
        # The Simulated Annealing algorithm will now use this smoothed value as its energy
        "total_relocated_mean": adjusted_mean_relocated,
    }
    return results


def get_neighbor(beds_A, beds_B, beds_C, total_beds):
    while True:
        state = {"A": beds_A, "B": beds_B, "C": beds_C}
        giver, receiver = random.sample(["A", "B", "C"], 2)

        state[giver] -= 1
        state[receiver] += 1

        # Enforce the baseline constraints defined in the problem
        if state["A"] >= 0 and state["B"] >= 0 and state["C"] >= 0:
            return state["A"], state["B"], state["C"]


def simulated_annealing_optimization(total_beds, max_steps=100, iterations_per_eval=10):
    # Initialize with a safe, valid baseline state
    current_A = 30
    current_B = 15
    current_C = total_beds - current_A - current_B

    current_state = (current_A, current_B, current_C)

    # Evaluate initial state energy (Energy = total relocated patients)
    current_res = run_multiple_simulations(
        *current_state, iterations=iterations_per_eval
    )
    current_energy = current_res["total_relocated_mean"]

    best_state = current_state
    best_energy = current_energy

    for k in range(max_steps):
        # Temperature Scheme: T = sqrt(1 / (1 + k))
        T = math.sqrt(1.0 / (1.0 + k))

        # Propose a neighbor state
        next_A, next_B, next_C = get_neighbor(*current_state, total_beds)

        # Evaluate the neighbor state
        next_res = run_multiple_simulations(
            next_A, next_B, next_C, iterations=iterations_per_eval
        )
        next_energy = next_res["total_relocated_mean"]

        # Calculate change in energy
        delta_E = next_energy - current_energy

        # Acceptance Criteria
        if delta_E < 0:
            # Improvement: Accept automatically
            current_state = (next_A, next_B, next_C)
            current_energy = next_energy

            # Track the global best found so far
            if next_energy < best_energy:
                best_state = current_state
                best_energy = next_energy
        else:
            # Worse state: Accept conditionally based on temperature and magnitude of worsening
            # Note: Division by T handles the cooling scheme mathematically.
            # If T is very small, the probability of accepting a worse state approaches 0.
            probability = math.exp(-delta_E / T)
            if random.random() < probability:
                current_state = (next_A, next_B, next_C)
                current_energy = next_energy

    return best_state, best_energy


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    print("--- Primary Performance Measures (Baseline: 30, 15, 30) ---")
    baseline = run_multiple_simulations(30, 15, 30, iterations=50)
    print(f"Prob all beds occupied on arrival: {baseline['prob_full_on_arrival']}")
    print(f"Mean relocated by type: {baseline['mean_relocated']}")
    print(f"Total mean relocated: {baseline['total_relocated_mean']}")
    print(f"Mean bed utilization: {baseline['mean_utilization']}\n")

    print("--- Sensitivity: Optimal Distribution for 75 beds ---")
    print("--- Optimization: Simulated Annealing (75 beds) ---")
    # Using 100 steps. Due to simulation noise, SA may require higher iterations per eval to stabilize.
    opt_dist, min_reloc = simulated_annealing_optimization(
        75, max_steps=100, iterations_per_eval=15
    )
    print(
        f"Optimal distribution found: {opt_dist} with ~{min_reloc:.2f} relocated patients\n"
    )
    optimized = run_multiple_simulations(*opt_dist, iterations=50)
    print(f"Prob all beds occupied on arrival: {optimized['prob_full_on_arrival']}")
    print(f"Mean relocated by type: {optimized['mean_relocated']}")
    print(f"Total mean relocated: {optimized['total_relocated_mean']}")
    print(f"Mean bed utilization: {optimized['mean_utilization']}\n")
    print(
        f"SA Optimal (A, B, C): {opt_dist} yielding ~{min_reloc:.2f} relocated patients\n"
    )

    print("--- Sensitivity: Exponential LOS Distribution (Using Optimal Beds) ---")
    exp = run_multiple_simulations(*opt_dist, iterations=50, use_exponential=True)
    print(f"Prob all beds occupied on arrival: {exp['prob_full_on_arrival']}")
    print(f"Mean relocated by type: {exp['mean_relocated']}")
    print(f"Total mean relocated: {exp['total_relocated_mean']}")
    print(f"Mean bed utilization: {exp['mean_utilization']}\n")

    print("--- Sensitivity: Changing Total Bed Capacity ---")
    for cap in [60, 80, 100]:
        print(f"Finding optimal for {cap} beds...")
        opt_c, min_r = simulated_annealing_optimization(
            cap, max_steps=100, iterations_per_eval=10
        )
        print(f"Capacity {cap}: Optimal {opt_c}, Relocated: {min_r:.2f}")
