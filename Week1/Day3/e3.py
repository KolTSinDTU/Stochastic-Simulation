import matplotlib.pyplot as plt
import random
import math
import numpy as np
import heapq

SAMPLES = 10_000

def get_exponential(rate):
    return -math.log(random.random()) / rate

def get_erlang(mean, k):
    return random.gammavariate(k, mean / k)

def get_hyperexponential(p1=0.8, lambda1=0.8333, lambda2=5.0):
    if random.random() < p1:
        return random.expovariate(lambda1)
    else:
        return random.expovariate(lambda2)

def get_pareto(mean, k):
    xm = mean * (k - 1) / k
    return xm * random.paretovariate(k)

def run_simulation(m, arrival_rate, service_time_func, total_customers):
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
                # Schedule departure using provided service time generator
                service_duration = service_time_func()
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

if __name__ == "__main__":
    # Configure simulation parameters
    num_servers = 10
    arrival_rate = 1
    mean_service_time = 8.0

    # Define scenarios with different service time generators
    scenarios = [
        ("Constant", lambda: mean_service_time),
        ("Erlang-1.05", lambda: get_erlang(mean_service_time, 1.05)),
        ("Erlang-2.05", lambda: get_erlang(mean_service_time, 2.05)),
        ("Pareto-1.05", lambda: get_pareto(mean_service_time, 1.05)),
        ("Exponential", lambda: get_exponential(1/mean_service_time))
    ]

    # Execute simulations
    blocking_results = []
    labels = []
    
    print(f"--- Starting Service Distribution Comparison ---")
    for label, service_f in scenarios:
        fraction = run_simulation(num_servers, arrival_rate, service_f, SAMPLES)
        blocking_results.append(fraction)
        labels.append(label)
        print(f"{label}: {fraction:.4f}")

    # Visualize results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2c3e50', '#34495e', '#7f8c8d', '#e74c3c']
    bars = ax.bar(labels, blocking_results, color=colors, alpha=0.8)

    ax.set_ylabel("Fraction of Blocked Customers")
    ax.set_title("--- Results: Comparison of Service Distributions ---")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.001,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()
    plt.show()
