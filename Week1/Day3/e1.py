import matplotlib.pyplot as plt
import random as rand
import math
import numpy as np
from queue import PriorityQueue
from scipy import stats

SAMPLES = 10_000


def exponential_distribution(lam):
    # 1. Generate a uniform random number
    u = rand.random()
    # 2. Apply inverse transform for exponential distribution
    return -math.log(u) / lam


class EventType:
    CustomerArrival = 1
    ServiceCompletion = 2


class Simulation:
    def __init__(self, num_servers, service_rate, arrival_rate):
        self.num_servers = num_servers
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.current_time = 0
        self.event_queue = PriorityQueue()
        self.customers_blocked = 0
        self.customers_arrived = 0

    def schedule_event(self, event_time, event_type):
        # 1. Add event to priority queue
        self.event_queue.put((event_time, event_type))

    def add_customer_arrival(self, arrival_time=None):
        # 1. Determine arrival time
        if arrival_time is None:
            arrival_time = exponential_distribution(self.arrival_rate)
        # 2. Schedule arrival event
        self.schedule_event(arrival_time, EventType.CustomerArrival)

    def process_event(self):
        # 1. Retrieve next event
        event_time, event_type = self.event_queue.get()
        self.current_time = event_time

        # 2. Handle event based on type
        if event_type == EventType.CustomerArrival:
            if self.num_servers > 0:
                service_duration = exponential_distribution(self.service_rate)
                self.schedule_event(
                    self.current_time + service_duration, EventType.ServiceCompletion
                )
                self.num_servers -= 1
            else:
                self.customers_blocked += 1

        elif event_type == EventType.ServiceCompletion:
            self.num_servers += 1

    def get_blocking_fraction(self):
        # 1. Calculate and return blocking fraction
        return float(self.customers_blocked) / float(self.customers_arrived)

    def run_simulation(self):
        # 1. Generate arrival events
        previous_arrival_time = 0
        for _ in range(SAMPLES):
            inter_arrival_time = exponential_distribution(self.arrival_rate)
            self.add_customer_arrival(previous_arrival_time + inter_arrival_time)
            previous_arrival_time += inter_arrival_time
            self.customers_arrived += 1

        # 2. Process all events in queue
        while not self.event_queue.empty():
            self.process_event()


def analytical_blocking_fraction(num_servers, arrival_rate, mean_service_time):
    # 1. Calculate offered load (A)
    offered_load = arrival_rate * mean_service_time
    # 2. Calculate Erlang-B formula
    numerator = offered_load**num_servers / math.factorial(num_servers)
    denominator = sum((offered_load**k) / math.factorial(k) for k in range(num_servers + 1))
    return numerator / denominator


def calculate_confidence_interval(data, confidence=0.95):
    # 1. Calculate statistics
    num_observations = len(data)
    alpha_level = 1 - confidence
    mean_val = np.mean(data)
    std_deviation = np.std(data, ddof=1)

    # 2. Calculate critical value and margin of error
    t_critical = stats.t.ppf(1 - alpha_level / 2, df=num_observations - 1)
    margin_of_error = t_critical * (std_deviation / math.sqrt(num_observations))

    # 3. Return confidence interval bounds
    return float(mean_val - margin_of_error), float(mean_val + margin_of_error)


if __name__ == "__main__":
    # 1. Configure simulation parameters
    num_servers = 10
    service_rate = 1 / 8
    arrival_rate = 1
    num_runs = 10

    # 2. Execute multiple simulation runs
    blocking_fractions = []
    for _ in range(num_runs):
        sim = Simulation(num_servers, service_rate, arrival_rate)
        sim.run_simulation()
        blocking_fractions.append(sim.get_blocking_fraction())

    # 3. Visualize results
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

    # 4. Print statistical summary
    confidence_int = calculate_confidence_interval(blocking_fractions)
    print(f"--- Results ---")
    print(f"Confidence Interval (95%): {confidence_int}")
    print(f"Estimated Blocking Fraction: {np.mean(blocking_fractions):.4f}")
    print(f"Estimated Variance: {np.var(blocking_fractions, ddof=1):.8f}")
    print(f"Analytical Blocking Fraction: {theoretical_val:.4f}")

    plt.tight_layout()
    plt.show()
