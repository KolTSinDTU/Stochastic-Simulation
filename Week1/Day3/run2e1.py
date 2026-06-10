import random
import heapq
import numpy as np
import math

def run_simulation(m, mean_service_time, mean_interarrival_time, total_customers):
    # Event types
    ARRIVAL = 0
    DEPARTURE = 1
    
    # State variables
    busy_servers = 0
    blocked_customers = 0
    customers_arrived = 0
    
    # Event list (priority queue)
    # Each entry: (time, type)
    event_list = []
    
    # Schedule first arrival
    current_time = 0.0
    first_arrival = random.expovariate(1.0 / mean_interarrival_time)
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
                service_time = random.expovariate(1.0 / mean_service_time)
                heapq.heappush(event_list, (current_time + service_time, DEPARTURE))
            else:
                # Blocked
                blocked_customers += 1
                
            # Schedule next arrival if we haven't reached the limit
            if customers_arrived < total_customers:
                interarrival_time = random.expovariate(1.0 / mean_interarrival_time)
                heapq.heappush(event_list, (current_time + interarrival_time, ARRIVAL))
                
        elif event_type == DEPARTURE:
            busy_servers -= 1
            
    return blocked_customers / total_customers

def main():
    # Parameters
    m = 10
    mean_service_time = 8.0
    mean_interarrival_time = 1.0
    customers_per_replication = 10000
    num_replications = 10
    
    blocking_fractions = []
    
    print(f"Running {num_replications} replications of {customers_per_replication} customers each...")
    
    for i in range(num_replications):
        fraction = run_simulation(m, mean_service_time, mean_interarrival_time, customers_per_replication)
        blocking_fractions.append(fraction)
        print(f"  Replication {i+1}: Blocked fraction = {fraction:.4f}")
        
    # Statistical analysis
    mean_blocking = np.mean(blocking_fractions)
    std_blocking = np.std(blocking_fractions, ddof=1)
    
    # 95% Confidence Interval using t-distribution for N=10 (df=9)
    # t value for 95% CI and 9 degrees of freedom is approx 2.262
    t_critical = 2.262 
    margin_of_error = t_critical * (std_blocking / math.sqrt(num_replications))
    
    print("\n--- Results ---")
    print(f"Mean Fraction of Blocked Customers: {mean_blocking:.4f}")
    print(f"95% Confidence Interval: [{mean_blocking - margin_of_error:.4f}, {mean_blocking + margin_of_error:.4f}]")
    
    # Traffic A = lambda * E[S]
    traffic_A = (1.0 / mean_interarrival_time) * mean_service_time
    print(f"Traffic Intensity (A): {traffic_A}")

if __name__ == "__main__":
    main()
