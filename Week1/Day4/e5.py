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


class EventType:
    CustomerArrival = 1
    ServiceCompletion = 2


class Simulation:
    def __init__(self, m, service_rate, arrival_rate):
        self.m = m
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.current_time = 0
        self.event_queue = PriorityQueue()
        self.customers_blocked = 0
        self.customers_arrived = 0
        self.service_times = []
        self.inter_arrival_times = []

    def schedule_event(self, event_time, event_type):
        self.event_queue.put((event_time, event_type))

    def add_customer_arrival(self, arrival_time=None):
        if arrival_time is None:
            arrival_time = exponential_distribution(self.arrival_rate)
        self.schedule_event(arrival_time, EventType.CustomerArrival)
        self.inter_arrival_times.append(arrival_time)

    def process_event(self):
        event_time, event_type = self.event_queue.get()
        self.current_time = event_time

        # print(
        #     f"Processing event at time {self.current_time:.2f}, type: {event_type} and m = {self.m}"
        # )

        if event_type == EventType.CustomerArrival:
            # Server available
            if self.m > 0:
                service_time = exponential_distribution(self.service_rate)
                self.schedule_event(
                    self.current_time + service_time, EventType.ServiceCompletion
                )
                self.service_times.append(service_time)
                self.m -= 1

            else:
                self.customers_blocked += 1

        elif event_type == EventType.ServiceCompletion:
            self.m += 1

    def get_blocking_fraction(self):
        return float(self.customers_blocked) / float(self.customers_arrived)

    def get_mean_service_time(self):
        return np.mean(self.service_times)

    def get_mean_inter_arrival_time(self):
        return np.mean(self.inter_arrival_times)

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


def get_covariance(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
    return covariance


if __name__ == "__main__":
    m = 10
    service_rate = 1 / 8
    arrival_rate = 1

    blocking_fractions = []
    avg_service_times = []
    avg_inter_arrival_times = []
    for _ in range(10):
        sim = Simulation(m, service_rate, arrival_rate)
        sim.run()
        blocking_fractions.append(sim.get_blocking_fraction())
        avg_service_times.append(sim.get_mean_service_time())
        avg_inter_arrival_times.append(sim.get_mean_inter_arrival_time())

    confidence_int = confidence_interval(blocking_fractions)
    print(
        f"Analytical Blocking Fraction: {analytical_blocking_fraction(m, arrival_rate, 1/service_rate)} \n"
    )
    print(f"Confidence interval for crude estimate: {confidence_int}")
    print(f"Variance: {np.var(blocking_fractions, ddof=1):.8f} \n")

    covariance_service = get_covariance(blocking_fractions, avg_service_times)
    var_service_times = np.var(avg_service_times, ddof=1)
    c = -covariance_service / var_service_times
    control_variate_estimates = [
        blocking_fractions[i] + c * (avg_service_times[i] - 8)
        for i in range(len(blocking_fractions))
    ]

    ci_lower, ci_upper = confidence_interval(control_variate_estimates)
    # print(f"Estimated Mean (Control Variate): {np.mean(control_variate_estimates):.4f}")
    print(
        f"Confidence Interval with service times as control variates: [{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    print(
        f"Variance with control variates: {np.var(control_variate_estimates, ddof=1):.8f} \n"
    )

    covariance_arrivals = get_covariance(blocking_fractions, avg_inter_arrival_times)
    var_inter_arrival_times = np.var(avg_inter_arrival_times, ddof=1)
    c = -covariance_arrivals / var_inter_arrival_times
    control_variate_estimates = [
        blocking_fractions[i] + c * (avg_inter_arrival_times[i] - 8)
        for i in range(len(blocking_fractions))
    ]

    print(
        f"Confidence Interval with inter-arrival times as control variates: [{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    print(
        f"Variance with control variates (inter-arrival times): {np.var(control_variate_estimates, ddof=1):.8f}"
    )
