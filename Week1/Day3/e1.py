import matplotlib.pyplot as plt
import random
import math
import numpy as np
import heapq
from scipy import stats

SAMPLES = 10_000

def get_exponential(rate):
    """Generate an exponential random variable."""
    return -math.log(random.random()) / rate

def run_simulation(m, service_rate, arrival_rate, total_customers):
    # Event types
    ARRIVAL = 0
    DEPARTURE = 1
    
    # State variables
    busy_servers = 0
    blocked_customers = 0
    customers_arrived = 0
    
    # Event list (priority queue)
    event_list = []
    
    # Schedule first arrival
    current_time = 0.0
    first_arrival = get_exponential(arrival_rate)
    heapq.heappush(event_list, (first_arrival, ARRIVAL))
    
    while customers_arrived < total_customers or event_list:
        if not event_list:
            break
            
        current_time, event_type = heapq.heappop(event_list)
        
        if event_type == ARRIVAL:
            customers_arrived += 1
            
            # Decide if customer is accepted or blocked
            if busy_servers < m:
                # Accepted
                busy_servers += 1
                # Schedule departure
                service_duration = get_exponential(service_rate)
                heapq.heappush(event_list, (current_time + service_duration, DEPARTURE))
            else:
                # Blocked
                blocked_customers += 1
                
            # Schedule next arrival
            if customers_arrived < total_customers:
                interarrival_time = get_exponential(arrival_rate)
                heapq.heappush(event_list, (current_time + interarrival_time, ARRIVAL))
                
        elif event_type == DEPARTURE:
            busy_servers -= 1
            
    return blocked_customers / total_customers

def analytical_blocking_fraction(num_servers, arrival_rate, mean_service_time):
    # Calculate offered load (A)
    offered_load = arrival_rate * mean_service_time
    # Calculate Erlang-B formula
    numerator = offered_load**num_servers / math.factorial(num_servers)
    denominator = sum((offered_load**k) / math.factorial(k) for k in range(num_servers + 1))
    return numerator / denominator

def calculate_confidence_interval(data, confidence=0.95):
    num_observations = len(data)
    alpha_level = 1 - confidence
    mean_val = np.mean(data)
    std_deviation = np.std(data, ddof=1)

    t_critical = stats.t.ppf(1 - alpha_level / 2, df=num_observations - 1)
    margin_of_error = t_critical * (std_deviation / math.sqrt(num_observations))

    return float(mean_val - margin_of_error), float(mean_val + margin_of_error)

if __name__ == "__main__":
    # Configure simulation parameters
    num_servers = 10
    service_rate = 1 / 8
    arrival_rate = 1
    num_runs = 10

    # Execute multiple simulation runs
    blocking_fractions = []
    for i in range(num_runs):
        fraction = run_simulation(num_servers, service_rate, arrival_rate, SAMPLES)
        blocking_fractions.append(fraction)
        print(f"Run {i+1}: {fraction:.4f}")

    # Visualize results
    run_indices = np.arange(1, num_runs + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        run_indices, blocking_fractions, marker="o", linestyle="-", color="#2c3e50", linewidth=2, label="Simulated"
    )

    theoretical_val = analytical_blocking_fraction(num_servers, arrival_rate, 1/service_rate)
    ax.axhline(
        y=theoretical_val,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label="Theoretical (Erlang-B)",
    )

    ax.set_xticks(run_indices)
    ax.set_xlabel("Simulation Run Index")
    ax.set_ylabel("Blocking Fraction")
    ax.set_title("--- Results: Blocking Fraction Simulation ---")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Print statistical summary
    confidence_int = calculate_confidence_interval(blocking_fractions)
    print(f"\n--- Results ---")
    print(f"Confidence Interval (95%): {confidence_int}")
    print(f"Estimated Blocking Fraction: {np.mean(blocking_fractions):.4f}")
    print(f"Estimated Variance: {np.var(blocking_fractions, ddof=1):.8f}")
    print(f"Analytical Blocking Fraction: {theoretical_val:.4f}")

    plt.tight_layout()
    plt.show()
