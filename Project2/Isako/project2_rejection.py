import math
import random
import heapq
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
REPLICATIONS = 100          # Number of replications per simulation run
DEFAULT_TOTAL_BEDS = 75     # Default total beds to optimize
LOS_DISTRIBUTION = 'lognormal'  # Default LOS distribution ('lognormal' or 'exponential')
PLOT_DIR = '.'              # Directory where plots will be saved

# ==============================================================================
# MODEL PARAMETERS & CONSTANTS
# ==============================================================================
# Peak time for Ward A and B arrivals is 182.5 days.
# Ward A (Regular): lambda_1(t) = -t^2 / 3650 + t / 10
# Ward B (Intensive): lambda_2(t) = 0.2 * lambda_1(t)
# Ward C (Other): lambda_3 = 6 (constant)

# Maximum rates on [0, 365] at t = 182.5
LAMBDA1_MAX = 9.125
LAMBDA2_MAX = 1.825

# Log-normal distribution parameters:
# Ward A (mean=8, std=8): mu = log(4*sqrt(2)), sigma^2 = log(2)
MU_A = math.log(4.0 * math.sqrt(2.0))
SIGMA_A = math.sqrt(math.log(2.0))

# Ward B (mean=12, std=12): mu = log(6*sqrt(2)), sigma^2 = log(2)
MU_B = math.log(6.0 * math.sqrt(2.0))
SIGMA_B = math.sqrt(math.log(2.0))

# Ward C (mean=10, std=10): mu = log(5*sqrt(2)), sigma^2 = log(2)
MU_C = math.log(5.0 * math.sqrt(2.0))
SIGMA_C = math.sqrt(math.log(2.0))

# ==============================================================================
# NHPP ARRIVAL GENERATION (ACCEPTANCE-REJECTION / THINNING METHOD)
# ==============================================================================
def lambda1_rate(t):
    """Arrival rate function for Ward A patients."""
    if 0 <= t <= 365:
        return - (1.0 / 3650.0) * (t ** 2) + 0.1 * t
    return 0.0

def generate_arrivals_rejection():
    """
    Generates arrival times for all three patient types on [0, 365]
    using the Acceptance-Rejection / Thinning method.
    Returns:
        List of tuples: (arrival_time, patient_type) sorted by arrival_time.
        Where patient_type is 1 (Ward A), 2 (Ward B), or 3 (Ward C).
    """
    arrivals = []
    
    # --- Ward A Arrivals ---
    # Generate homogeneous candidates at rate LAMBDA1_MAX and accept/reject them
    t = 0.0
    while True:
        t += -math.log(random.random()) / LAMBDA1_MAX
        if t > 365.0:
            break
        p_accept = lambda1_rate(t) / LAMBDA1_MAX
        if random.random() <= p_accept:
            arrivals.append((t, 1))
            
    # --- Ward B Arrivals ---
    # Generate homogeneous candidates at rate LAMBDA2_MAX and accept/reject them
    t = 0.0
    while True:
        t += -math.log(random.random()) / LAMBDA2_MAX
        if t > 365.0:
            break
        p_accept = lambda1_rate(t) / LAMBDA1_MAX
        if random.random() <= p_accept:
            arrivals.append((t, 2))
            
    # --- Ward C Arrivals ---
    # Ward C has constant arrival rate lambda_3 = 6.0
    t = 0.0
    while True:
        t += -math.log(random.random()) / 6.0  # Exponential(6) step
        if t > 365.0:
            break
        arrivals.append((t, 3))
        
    # Merge and sort all arrivals chronologically
    arrivals.sort(key=lambda x: x[0])
    return arrivals

# ==============================================================================
# LENGTH-OF-STAY SAMPLING
# ==============================================================================
def sample_los(ward, dist_type):
    """Samples length of stay for a given ward and distribution type."""
    if dist_type == 'lognormal':
        if ward == 'A':
            return np.random.lognormal(MU_A, SIGMA_A)
        elif ward == 'B':
            return np.random.lognormal(MU_B, SIGMA_B)
        elif ward == 'C':
            return np.random.lognormal(MU_C, SIGMA_C)
    elif dist_type == 'exponential':
        if ward == 'A':
            return random.expovariate(1.0 / 8.0)
        elif ward == 'B':
            return random.expovariate(1.0 / 12.0)
        elif ward == 'C':
            return random.expovariate(1.0 / 10.0)
    else:
        raise ValueError("Unknown distribution type")

# ==============================================================================
# DISCRETE EVENT SIMULATION ENGINE
# ==============================================================================
def run_simulation(N_A, N_B, N_C, arrivals, los_distribution="lognormal"):
    """
    Simulates patient flow for the given bed distribution and arrivals.
    """
    occupied_A = 0
    occupied_B = 0
    occupied_C = 0
    
    departures = []
    arrived_counts = [0, 0, 0]
    blocked_counts = [0, 0, 0]
    
    area_occupied_A = 0.0
    area_occupied_B = 0.0
    area_occupied_C = 0.0
    last_time = 0.0
    
    T = 365.0
    
    for arr_time, p_type in arrivals:
        # 1. Process departures that happen before the current arrival time
        while departures and departures[0][0] <= arr_time:
            dep_time, ward = heapq.heappop(departures)
            dt = dep_time - last_time
            area_occupied_A += occupied_A * dt
            area_occupied_B += occupied_B * dt
            area_occupied_C += occupied_C * dt
            last_time = dep_time
            
            if ward == 'A':
                occupied_A -= 1
            elif ward == 'B':
                occupied_B -= 1
            elif ward == 'C':
                occupied_C -= 1
                
        # 2. Integrate occupied bed area from last event to current arrival
        dt = arr_time - last_time
        area_occupied_A += occupied_A * dt
        area_occupied_B += occupied_B * dt
        area_occupied_C += occupied_C * dt
        last_time = arr_time
        
        # 3. Process the arrival
        idx = p_type - 1
        arrived_counts[idx] += 1
        
        if p_type == 1:  # Regular care (Ward A)
            if occupied_A < N_A:
                occupied_A += 1
                los = sample_los('A', los_distribution)
                heapq.heappush(departures, (arr_time + los, 'A'))
            else:
                blocked_counts[idx] += 1
                
        elif p_type == 2:  # Intensive care (Ward B)
            if occupied_B < N_B:
                occupied_B += 1
                los = sample_los('B', los_distribution)
                heapq.heappush(departures, (arr_time + los, 'B'))
            elif occupied_A < N_A:  # Overflow spillover to Ward A
                occupied_A += 1
                los = sample_los('B', los_distribution)
                heapq.heappush(departures, (arr_time + los, 'A'))
            else:
                blocked_counts[idx] += 1
                
        elif p_type == 3:  # Other care (Ward C)
            if occupied_C < N_C:
                occupied_C += 1
                los = sample_los('C', los_distribution)
                heapq.heappush(departures, (arr_time + los, 'C'))
            else:
                blocked_counts[idx] += 1
                
    # 4. Process remaining departures up to T = 365
    while departures and departures[0][0] <= T:
        dep_time, ward = heapq.heappop(departures)
        dt = dep_time - last_time
        area_occupied_A += occupied_A * dt
        area_occupied_B += occupied_B * dt
        area_occupied_C += occupied_C * dt
        last_time = dep_time
        
        if ward == 'A':
            occupied_A -= 1
        elif ward == 'B':
            occupied_B -= 1
        elif ward == 'C':
            occupied_C -= 1
            
    # 5. Final integration
    if last_time < T:
        dt = T - last_time
        area_occupied_A += occupied_A * dt
        area_occupied_B += occupied_B * dt
        area_occupied_C += occupied_C * dt
        
    util_A = (area_occupied_A / (T * N_A)) if N_A > 0 else 0.0
    util_B = (area_occupied_B / (T * N_B)) if N_B > 0 else 0.0
    util_C = (area_occupied_C / (T * N_C)) if N_C > 0 else 0.0
    
    return arrived_counts, blocked_counts, [util_A, util_B, util_C]

# ==============================================================================
# HIERARCHICAL OPTIMIZATION ENGINE
# ==============================================================================
def run_multiple_simulations(N_A, N_B, N_C, arrivals_list, los_dist):
    """Runs simulation multiple times on pre-generated arrivals and averages results."""
    total_arrived = np.zeros(3)
    total_blocked = np.zeros(3)
    total_utils = np.zeros(3)
    reps = len(arrivals_list)
    
    for r in range(reps):
        arrived, blocked, utils = run_simulation(N_A, N_B, N_C, arrivals_list[r], los_dist)
        total_arrived += arrived
        total_blocked += blocked
        total_utils += utils
        
    mean_arrived = total_arrived / reps
    mean_blocked = total_blocked / reps
    mean_utils = total_utils / reps
    
    blocking_probs = [mean_blocked[i] / mean_arrived[i] if mean_arrived[i] > 0 else 0.0 for i in range(3)]
    
    return mean_arrived, mean_blocked, blocking_probs, mean_utils

def optimize_bed_allocation_grid(total_beds, los_dist, coarse_arrivals, fine_arrivals):
    """
    Finds the optimal allocation of beds (N_A, N_B, N_C) that minimizes the 
    total number of relocated patients using a Two-Step Grid Search.
    """
    # Step 1: Coarse Grid Search (Step Size = 3)
    coarse_allocations = []
    for NA in range(1, total_beds - 1):
        for NB in range(1, total_beds - NA):
            NC = total_beds - NA - NB
            if NA % 3 == 0 and NB % 3 == 0:
                coarse_allocations.append((NA, NB, NC))
                
    if not coarse_allocations:
        for NA in range(1, total_beds - 1):
            for NB in range(1, total_beds - NA):
                NC = total_beds - NA - NB
                coarse_allocations.append((NA, NB, NC))
                
    best_coarse_alloc = None
    best_coarse_score = float('inf')
    
    for NA, NB, NC in coarse_allocations:
        _, blocked, _, _ = run_multiple_simulations(NA, NB, NC, coarse_arrivals, los_dist)
        score = sum(blocked)
        if score < best_coarse_score:
            best_coarse_score = score
            best_coarse_alloc = (NA, NB, NC)
            
    # Step 2: Fine Local Search (Step Size = 1) around the best coarse allocation
    c_NA, c_NB, c_NC = best_coarse_alloc
    fine_allocations = []
    for NA in range(max(1, c_NA - 3), min(total_beds - 1, c_NA + 4)):
        for NB in range(max(1, c_NB - 3), min(total_beds - NA, c_NB + 4)):
            NC = total_beds - NA - NB
            if NC >= 1:
                fine_allocations.append((NA, NB, NC))
                
    best_fine_alloc = best_coarse_alloc
    best_fine_score = best_coarse_score
    
    for NA, NB, NC in fine_allocations:
        _, blocked, _, _ = run_multiple_simulations(NA, NB, NC, fine_arrivals, los_dist)
        score = sum(blocked)
        if score < best_fine_score:
            best_fine_score = score
            best_fine_alloc = (NA, NB, NC)
            
    return best_fine_alloc, best_fine_score

def optimize_bed_allocation_sa(total_beds, los_dist, arrivals_list, init_temp=100.0, cooling_rate=0.95, min_temp=0.1, steps_per_temp=10):
    """
    Finds the optimal allocation of beds (N_A, N_B, N_C) using Simulated Annealing.
    Minimizes the total number of relocated patients.
    """
    # Initial configuration: equal distribution
    n_A = total_beds // 3
    n_B = total_beds // 3
    n_C = total_beds - n_A - n_B
    
    current_alloc = (n_A, n_B, n_C)
    _, blocked, _, _ = run_multiple_simulations(n_A, n_B, n_C, arrivals_list, los_dist)
    current_score = sum(blocked)
    
    best_alloc = current_alloc
    best_score = current_score
    
    T = init_temp
    
    while T > min_temp:
        for _ in range(steps_per_temp):
            # Propose a neighbor by moving 1 bed from one ward to another
            wards = ['A', 'B', 'C']
            from_ward, to_ward = random.sample(wards, 2)
            
            n_A, n_B, n_C = current_alloc
            if from_ward == 'A' and n_A > 1:
                n_A -= 1
                if to_ward == 'B': n_B += 1
                else: n_C += 1
            elif from_ward == 'B' and n_B > 1:
                n_B -= 1
                if to_ward == 'A': n_A += 1
                else: n_C += 1
            elif from_ward == 'C' and n_C > 1:
                n_C -= 1
                if to_ward == 'A': n_A += 1
                else: n_B += 1
            else:
                continue
            
            neighbor_alloc = (n_A, n_B, n_C)
            _, blocked, _, _ = run_multiple_simulations(n_A, n_B, n_C, arrivals_list, los_dist)
            neighbor_score = sum(blocked)
            
            diff = neighbor_score - current_score
            if diff < 0:
                accept = True
            else:
                # Boltzmann probability
                prob = math.exp(-diff / T)
                accept = random.random() < prob
                
            if accept:
                current_alloc = neighbor_alloc
                current_score = neighbor_score
                if current_score < best_score:
                    best_score = current_score
                    best_alloc = current_alloc
                    
        T *= cooling_rate
        
    return best_alloc, best_score

def optimize_bed_allocation(total_beds, los_dist, coarse_arrivals, fine_arrivals):
    """
    Finds the optimal bed allocation using Simulated Annealing, and compares it with Grid Search.
    """
    print(f"Optimizing allocation for {total_beds} beds ({los_dist} LOS)...")
    
    # 1. Run Simulated Annealing (using coarse_arrivals for speed)
    sa_alloc, sa_score_coarse = optimize_bed_allocation_sa(total_beds, los_dist, coarse_arrivals)
    # Evaluate SA best on fine_arrivals
    _, sa_blocked_fine, _, _ = run_multiple_simulations(*sa_alloc, fine_arrivals, los_dist)
    sa_score_fine = sum(sa_blocked_fine)
    
    # 2. Run Grid Search for comparison
    grid_alloc, grid_score_fine = optimize_bed_allocation_grid(total_beds, los_dist, coarse_arrivals, fine_arrivals)
    
    print(f"  [Optimizer Comparison for N={total_beds}]:")
    print(f"    - Simulated Annealing found: {sa_alloc} (Mean Relocated: {sa_score_fine:.2f})")
    print(f"    - Two-Step Grid Search found: {grid_alloc} (Mean Relocated: {grid_score_fine:.2f})")
    
    # Return the Simulated Annealing result to align with user choice
    return sa_alloc, sa_score_fine

# ==============================================================================
# REPORT PLOTTING FUNCTIONS
# ==============================================================================
def plot_arrival_rates():
    """Plots the arrival intensity curves for the three wards."""
    t_vals = np.linspace(0, 365, 500)
    lam1_vals = [lambda1_rate(t) for t in t_vals]
    lam2_vals = [0.2 * l1 for l1 in lam1_vals]
    lam3_vals = [6.0 for _ in t_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, lam1_vals, label="Ward A (Regular care)", color='#1abc9c', linewidth=2.5)
    plt.plot(t_vals, lam2_vals, label="Ward B (Intensive care)", color='#e67e22', linewidth=2.5)
    plt.plot(t_vals, lam3_vals, label="Ward C (Other care)", color='#3498db', linewidth=2.5, linestyle="--")
    
    plt.title("Patient Arrival Intensities Over the Epidemic Year", fontsize=14, pad=15)
    plt.xlabel("Time $t$ (days)", fontsize=12)
    plt.ylabel("Arrival Rate (patients/day)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/rejection_arrival_rates.png", dpi=300)
    plt.close()

def plot_allocation_heatmap(total_beds, los_dist, coarse_arrivals, fine_arrivals):
    """
    Runs simulations on a grid of bed allocations to create an optimization 
    heatmap for Ward A and Ward B beds.
    """
    print("Generating bed allocation optimization heatmap...")
    x_NA = np.arange(10, total_beds - 15, 2)
    y_NB = np.arange(2, 25, 2)
    
    Z = np.zeros((len(y_NB), len(x_NA)))
    
    for i, NB in enumerate(y_NB):
        for j, NA in enumerate(x_NA):
            NC = total_beds - NA - NB
            if NC >= 1:
                _, blocked, _, _ = run_multiple_simulations(NA, NB, NC, coarse_arrivals, los_dist)
                Z[i, j] = sum(blocked)
            else:
                Z[i, j] = np.nan
                
    plt.figure(figsize=(10, 7))
    plt.contourf(x_NA, y_NB, Z, 20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label("Mean Total Relocated Patients / Year", fontsize=11)
    
    plt.title(f"Optimization Heatmap ({total_beds} Total Beds): Relocations vs. (NA, NB) Bed Distribution", fontsize=13, pad=15)
    plt.xlabel("Ward A Beds ($N_A$)", fontsize=12)
    plt.ylabel("Ward B Beds ($N_B$)", fontsize=12)
    
    best_alloc, best_score = optimize_bed_allocation(total_beds, los_dist, coarse_arrivals, fine_arrivals)
    plt.scatter(best_alloc[0], best_alloc[1], color='#e74c3c', s=120, edgecolors='white', zorder=10, 
                label=f"Optimal Allocation: {best_alloc}\n(Relocated: {best_score:.1f})")
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/rejection_optimal_allocation_heatmap.png", dpi=300)
    plt.close()

def plot_los_sensitivity(best_alloc, coarse_arrivals, fine_arrivals):
    """Compares performance measures of the optimal allocation between Log-normal and Exponential LOS."""
    print("Comparing LOS distribution sensitivities...")
    NA, NB, NC = best_alloc
    
    _, _, bp_log, utils_log = run_multiple_simulations(NA, NB, NC, fine_arrivals, 'lognormal')
    _, _, bp_exp, utils_exp = run_multiple_simulations(NA, NB, NC, fine_arrivals, 'exponential')
    
    labels = ['Ward A', 'Ward B', 'Ward C']
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rects1 = ax.bar(x - width/2, bp_log, width, label='Log-normal', color='#2ecc71')
    rects2 = ax.bar(x + width/2, bp_exp, width, label='Exponential', color='#e74c3c')
    
    ax.set_ylabel('Blocking Probability on Arrival', fontsize=12)
    ax.set_title(f'Sensitivity analysis: Blocking Probabilities under optimal allocation {best_alloc}', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/rejection_los_sensitivity.png", dpi=300)
    plt.close()

def plot_capacity_impact(capacities, reps, los_dist):
    """Evaluates the optimal relocation numbers as a function of total capacity."""
    print("Evaluating capacity impacts...")
    results = []
    allocations = []
    
    # Pre-generate arrivals for capacity tests (reused for all searches in capacity)
    coarse_arrivals = [generate_arrivals_rejection() for _ in range(reps)]
    fine_arrivals = [generate_arrivals_rejection() for _ in range(max(reps * 2, 50))]
    
    for cap in capacities:
        best_alloc, score = optimize_bed_allocation(cap, los_dist, coarse_arrivals, fine_arrivals)
        results.append(score)
        allocations.append(best_alloc)
        
    plt.figure(figsize=(9, 5.5))
    plt.plot(capacities, results, marker='o', color='#8e44ad', linewidth=2.5, markersize=8)
    
    for i, txt in enumerate(allocations):
        plt.annotate(f"  Opt: {txt}\n  Reloc: {results[i]:.1f}", 
                     (capacities[i], results[i]), 
                     textcoords="offset points", 
                     xytext=(-15, 10), 
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        
    plt.title("Impact of Total Bed Capacity on Mean Relocated Patients", fontsize=13, pad=15)
    plt.xlabel("Total Beds in System ($N$)", fontsize=12)
    plt.ylabel("Mean Total Relocated Patients / Year", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/rejection_capacity_impact.png", dpi=300)
    plt.close()

# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================
def main():
    print("======================================================================")
    print("STOCHASTIC HOSPITAL BED SIMULATION MODEL")
    print("METHOD: ACCEPTANCE-REJECTION / THINNING (OPTION B)")
    print(f"REPLICATIONS PER CONFIGURATION: {REPLICATIONS}")
    print("======================================================================")
    
    # Generate static arrival datasets once
    print("Pre-generating arrival datasets...")
    coarse_arrivals = [generate_arrivals_rejection() for _ in range(REPLICATIONS)]
    fine_arrivals = [generate_arrivals_rejection() for _ in range(max(REPLICATIONS * 2, 100))]
    
    # 1. Plot arrivals rate curves
    plot_arrival_rates()
    print("Arrival intensity curve plot generated.")
    
    # 2. Simulate default allocation to display baseline performance
    baseline_alloc = (25, 15, 35)
    print(f"\nRunning Baseline Simulation: Beds A={baseline_alloc[0]}, B={baseline_alloc[1]}, C={baseline_alloc[2]} (Log-normal LOS)")
    arrived, blocked, bp, utils = run_multiple_simulations(*baseline_alloc, coarse_arrivals, 'lognormal')
    
    print("\n--- Baseline Results (Lognormal LOS) ---")
    for i, name in enumerate(['Ward A (Regular)', 'Ward B (Intensive)', 'Ward C (Other)']):
        print(f"{name}:")
        print(f"  Mean Arrivals: {arrived[i]:.2f}")
        print(f"  Mean Relocated: {blocked[i]:.2f}")
        print(f"  Blocking Probability: {bp[i]:.4f}")
        print(f"  Bed Utilization: {utils[i]:.4f}")
    print(f"Total Mean Relocated: {sum(blocked):.2f}")
    
    # 3. Optimize bed allocation for N = 75
    best_alloc, best_score = optimize_bed_allocation(DEFAULT_TOTAL_BEDS, 'lognormal', coarse_arrivals, fine_arrivals)
    print(f"\nOptimal Bed Distribution: NA={best_alloc[0]}, NB={best_alloc[1]}, NC={best_alloc[2]}")
    print(f"Minimum Total Relocated Patients: {best_score:.2f}")
    
    # Get detailed metrics for optimal allocation
    opt_arr, opt_bl, opt_bp, opt_ut = run_multiple_simulations(*best_alloc, fine_arrivals, 'lognormal')
    print("\n--- Optimal Bed Distribution Results (Lognormal LOS) ---")
    for i, name in enumerate(['Ward A (Regular)', 'Ward B (Intensive)', 'Ward C (Other)']):
        print(f"{name}:")
        print(f"  Mean Arrivals: {opt_arr[i]:.2f}")
        print(f"  Mean Relocated: {opt_bl[i]:.2f}")
        print(f"  Blocking Probability: {opt_bp[i]:.4f}")
        print(f"  Bed Utilization: {opt_ut[i]:.4f}")
    print(f"Total Mean Relocated (Sum): {sum(opt_bl):.2f}")
    
    # 4. Generate optimization heatmap
    plot_allocation_heatmap(DEFAULT_TOTAL_BEDS, 'lognormal', coarse_arrivals, fine_arrivals)
    
    # 5. Sensitivity to Length of Stay (LOS) distribution
    plot_los_sensitivity(best_alloc, coarse_arrivals, fine_arrivals)
    
    # 6. Evaluate impact of total beds (Capacity analysis)
    capacities_to_test = [60, 75, 80, 100]
    plot_capacity_impact(capacities_to_test, REPLICATIONS, 'lognormal')
    
    print("\n======================================================================")
    print("Simulation analysis complete!")
    print(f"All plots successfully saved to: {PLOT_DIR}/")
    print("======================================================================")

if __name__ == "__main__":
    main()
