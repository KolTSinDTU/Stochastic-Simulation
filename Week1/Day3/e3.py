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


def get_erlang(mean=1.0, k=1):
    # 1. Generate Erlang distributed value using gamma distribution
    return rand.gammavariate(k, mean / k)


def get_hyperexponential(p1=0.8, lambda1=0.8333, lambda2=5.0):
    # 1. Randomly select between two exponential distributions
    if rand.random() < p1:
        return rand.expovariate(lambda1)
    else:
        return rand.expovariate(lambda2)


class EventType:
    CustomerArrival = 1
    ServiceCompletion = 2


class DistributionType:
    Exponential = 1
    Erlang = 2
    Hyperexponential = 3
    Constant = 4


class Simulation:
    def __init__(
        self,
        num_servers,
        service_rate,
        arrival_rate,
        distribution_type=DistributionType.Exponential,
        k_parameter=4,
    ):
        self.num_servers = num_servers
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.distribution_type = distribution_type
        self.k_parameter = k_parameter
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
                if self.distribution_type == DistributionType.Exponential:
                    service_duration = exponential_distribution(self.service_rate)
                elif self.distribution_type == DistributionType.Erlang:
                    service_duration = get_erlang(mean=8.0, k=self.k_parameter)
                elif self.distribution_type == DistributionType.Hyperexponential:
                    service_duration = get_hyperexponential()
                elif self.distribution_type == DistributionType.Constant:
                    service_duration = 8.0

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


if __name__ == "__main__":
    # 1. Configure simulation parameters
    num_servers = 10
    service_rate = 1 / 8
    arrival_rate = 1

    # 2. Initialize simulations with different distributions
    blocking_results = []
    sim_configs = [
        Simulation(num_servers, service_rate, arrival_rate, DistributionType.Constant),
        Simulation(num_servers, service_rate, arrival_rate, DistributionType.Erlang, k_parameter=1.05),
        Simulation(num_servers, service_rate, arrival_rate, DistributionType.Erlang, k_parameter=2.05),
        Simulation(num_servers, service_rate, arrival_rate, DistributionType.Exponential)
    ]

    # 3. Execute simulations
    for simulation in sim_configs:
        simulation.run_simulation()
        blocking_results.append(simulation.get_blocking_fraction())

    # 4. Visualize results
    labels = ["Constant", "Erlang-1.05", "Erlang-2.05", "Exponential"]
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

    print(f"--- Simulation Complete ---")
    for label, fraction in zip(labels, blocking_results):
        print(f"{label}: {fraction:.4f}")

    plt.tight_layout()
    plt.show()
