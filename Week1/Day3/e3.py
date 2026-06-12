import matplotlib.pyplot as plt
import random as rand
import math
import numpy as np
from queue import PriorityQueue

from scipy import stats

SAMPLES = 10_000


def exponential_distribution(lam):
    u = rand.random()
    return -math.log(u) / lam


def get_erlang(mean=1.0, k=1):
    return rand.gammavariate(k, mean / k)


def get_hyperexponential(p1=0.8, lambda1=0.8333, lambda2=5.0):
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
        m,
        service_rate,
        arrival_rate,
        distribution_type=DistributionType.Exponential,
        k=4,
    ):
        self.m = m
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.distribution_type = distribution_type
        self.k = k
        self.current_time = 0
        self.event_queue = PriorityQueue()
        self.customers_blocked = 0
        self.customers_arrived = 0

    def schedule_event(self, event_time, event_type):
        self.event_queue.put((event_time, event_type))

    def add_customer_arrival(self, arrival_time=None):
        if arrival_time is None:
            arrival_time = exponential_distribution(self.arrival_rate)
        self.schedule_event(arrival_time, EventType.CustomerArrival)

    def process_event(self):
        event_time, event_type = self.event_queue.get()
        self.current_time = event_time

        # print(
        #     f"Processing event at time {self.current_time:.2f}, type: {event_type} and m = {self.m}"
        # )

        if event_type == EventType.CustomerArrival:
            # Server available
            if self.m > 0:
                if self.distribution_type == DistributionType.Exponential:
                    service_time = exponential_distribution(self.service_rate)
                elif self.distribution_type == DistributionType.Erlang:
                    service_time = get_erlang(mean=8.0, k=self.k)
                elif self.distribution_type == DistributionType.Hyperexponential:
                    service_time = get_hyperexponential()
                elif self.distribution_type == DistributionType.Constant:
                    service_time = 8.0

                self.schedule_event(
                    self.current_time + service_time, EventType.ServiceCompletion
                )
                self.m -= 1

            else:
                self.customers_blocked += 1

        elif event_type == EventType.ServiceCompletion:
            self.m += 1

    def get_blocking_fraction(self):
        return float(self.customers_blocked) / float(self.customers_arrived)

    def run(self):
        prev_time = 0
        for _ in range(SAMPLES):
            inter_arrival_time = exponential_distribution(self.arrival_rate)
            self.add_customer_arrival(prev_time + inter_arrival_time)
            prev_time += inter_arrival_time
            self.customers_arrived += 1

        while not self.event_queue.empty():
            self.process_event()


def analytical_blocking_fraction(m, lam, s):
    A = lam * s
    numerator = A**m / math.factorial(m)
    denominator = sum((A**k) / math.factorial(k) for k in range(m + 1))
    return numerator / denominator


def confidence_interval(data, confidence=0.95):
    n = len(data)
    alpha = 1 - confidence

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    # Calculate margin of error using the critical value, not the confidence percentage
    margin_of_error = t_critical * (std_dev / math.sqrt(n))

    lower_ci = mean - margin_of_error
    upper_ci = mean + margin_of_error
    return float(lower_ci), float(upper_ci)


if __name__ == "__main__":
    m = 10
    service_rate = 1 / 8
    arrival_rate = 1

    blocking_fractions = []
    simulation1 = Simulation(m, service_rate, arrival_rate, DistributionType.Constant)
    simulation2 = Simulation(
        m, service_rate, arrival_rate, DistributionType.Erlang, k=1.05
    )
    simulation3 = Simulation(
        m, service_rate, arrival_rate, DistributionType.Erlang, k=2.05
    )
    simulation4 = Simulation(
        m, service_rate, arrival_rate, DistributionType.Exponential
    )
    simulations = [simulation1, simulation2, simulation3, simulation4]

    for sim in simulations:
        sim.run()
        blocking_fractions.append(sim.get_blocking_fraction())

    labels = ["Constant", "Erlang-1.05", "Erlang-2.05", "Exponential"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        labels,
        blocking_fractions,
        capsize=10,
        color=["skyblue", "lightgreen", "salmon", "lightcoral"],
    )

    plt.ylabel("Fraction of Blocked Customers")
    plt.title("Comparison of Service Distributions (Mean Inter-arrival Time = 8.0)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.002,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()
