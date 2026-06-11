import random
import heapq
import numpy as np
import math
import matplotlib.pyplot as plt

def run_simulation(m, mean_service_time, arrival_func, total_customers):
    """
    Runs a discrete event simulation for an M/G/m/m queue.
    arrival_func: A callable that returns the next inter-arrival time.
    """
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
    first_arrival = arrival_func()
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
                # Schedule departure (Exponential service time, mean 8)
                service_time = random.expovariate(1.0 / mean_service_time)
                heapq.heappush(event_list, (current_time + service_time, DEPARTURE))
            else:
                # Blocked
                blocked_customers += 1
                
            # Schedule next arrival
            if customers_arrived < total_customers:
                interarrival_time = arrival_func()
                heapq.heappush(event_list, (current_time + interarrival_time, ARRIVAL))
                
        elif event_type == DEPARTURE:
            busy_servers -= 1
            
    return blocked_customers / total_customers

# Arrival Generators
def get_arrival_exponential(mean=1.0):
    return random.expovariate(1.0 / mean)

def get_arrival_erlang(mean=1.0, k=1):
    return random.gammavariate(k, mean / k)

def get_arrival_hyperexponential(p1=0.8, lambda1=0.8333, lambda2=5.0):
    if random.random() < p1:
        return random.expovariate(lambda1)
    else:
        return random.expovariate(lambda2)

def main():
    # Simulation Parameters
    m = 10
    mean_service_time = 8.0
    customers_per_replication = 10000
    num_replications = 10
    
    scenarios = [
        ("Exponential", lambda: get_arrival_exponential(1.0)),
        ("Erlang-4", lambda: get_arrival_erlang(1.0, 4)),
        ("Hyperexponential", get_arrival_hyperexponential)
    ]
    
    results_mean = []
    results_ci = []
    
    print(f"Starting experiments ({num_replications} replications per scenario)...")
    
    for name, arrival_f in scenarios:
        print(f"\nRunning Scenario: {name}")
        blocking_fractions = []
        for i in range(num_replications):
            fraction = run_simulation(m, mean_service_time, arrival_f, customers_per_replication)
            blocking_fractions.append(fraction)
            print(f"  Replication {i+1}: {fraction:.4f}")
            
        mean = np.mean(blocking_fractions)
        std = np.std(blocking_fractions, ddof=1)
        t_critical = 2.262 # for df=9, 95% CI
        margin_of_error = t_critical * (std / math.sqrt(num_replications))
        
        results_mean.append(mean)
        results_ci.append(margin_of_error)
        
        print(f"Result: {mean:.4f} +/- {margin_of_error:.4f}")

    # Visualization
    labels = [s[0] for s in scenarios]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, results_mean, yerr=results_ci, capsize=10, color=['skyblue', 'lightgreen', 'salmon'])
    
    plt.ylabel('Fraction of Blocked Customers')
    plt.title('Comparison of Arrival Distributions (Mean Inter-arrival Time = 1.0)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    # plot_path = 'Day3/arrival_comparison.png'
    # plt.savefig(plot_path)
    # print(f"\nSuccess! Chart saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
