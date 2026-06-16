import matplotlib.pyplot as plt
import random as rand
import math
import numpy as np
from queue import PriorityQueue
from scipy import stats

TOTAL_SAMPLES = 10_000


def generate_exponential_sample(lambda_param):
    # 1. Generate random uniform
    u = rand.random()
    # 2. Apply inverse transform
    return -math.log(u) / lambda_param


class EventType:
    CUSTOMER_ARRIVAL = 1
    SERVICE_COMPLETION = 2


class ErlangBSimulation:
    def __init__(self, num_servers, service_rate, arrival_rate):
        # 1. Initialize simulation state
        self.num_servers = num_servers
        self.available_servers = num_servers
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.current_time = 0
        self.event_queue = PriorityQueue()
        self.customers_blocked = 0
        self.customers_arrived = 0
        self.service_times = []
        self.inter_arrival_times = []

    def schedule_event(self, event_time, event_type):
        # 1. Add event to priority queue
        self.event_queue.put((event_time, event_type))

    def add_customer_arrival(self, arrival_time=None):
        # 1. Determine arrival time
        if arrival_time is None:
            arrival_time = generate_exponential_sample(self.arrival_rate)
        # 2. Schedule and log
        self.schedule_event(arrival_time, EventType.CUSTOMER_ARRIVAL)
        self.inter_arrival_times.append(arrival_time)

    def process_event(self):
        # 1. Get next event
        event_time, event_type = self.event_queue.get()
        self.current_time = event_time

        # 2. Handle arrival or completion
        if event_type == EventType.CUSTOMER_ARRIVAL:
            if self.available_servers > 0:
                service_duration = generate_exponential_sample(self.service_rate)
                self.schedule_event(
                    self.current_time + service_duration, EventType.SERVICE_COMPLETION
                )
                self.service_times.append(service_duration)
                self.available_servers -= 1
            else:
                self.customers_blocked += 1

        elif event_type == EventType.SERVICE_COMPLETION:
            self.available_servers += 1

    def get_blocking_fraction(self):
        # 1. Calculate ratio
        return float(self.customers_blocked) / float(self.customers_arrived)

    def get_mean_service_time(self):
        # 1. Return average service time
        return np.mean(self.service_times)

    def get_mean_inter_arrival_time(self):
        # 1. Return average inter-arrival time
        return np.mean(self.inter_arrival_times)

    def run_simulation(self):
        # 1. Setup arrivals
        previous_arrival_time = 0
        for _ in range(TOTAL_SAMPLES):
            inter_arrival_duration = generate_exponential_sample(self.arrival_rate)
            self.add_customer_arrival(previous_arrival_time + inter_arrival_duration)
            previous_arrival_time += inter_arrival_duration
            self.customers_arrived += 1

        # 2. Process all events
        while not self.event_queue.empty():
            self.process_event()


def calculate_analytical_blocking(num_servers, arrival_rate, mean_service_duration):
    # 1. Calculate offered load
    offered_load = arrival_rate * mean_service_duration
    # 2. Erlang B formula
    numerator = offered_load**num_servers / math.factorial(num_servers)
    denominator = sum((offered_load**k) / math.factorial(k) for k in range(num_servers + 1))
    return numerator / denominator


def calculate_confidence_interval(data_points, confidence_level=0.95):
    # 1. Statistical calculations
    n_samples = len(data_points)
    alpha_val = 1 - confidence_level
    sample_mean = np.mean(data_points)
    sample_std_dev = np.std(data_points, ddof=1)
    t_critical_val = stats.t.ppf(1 - alpha_val / 2, df=n_samples - 1)
    margin_of_error_val = t_critical_val * (sample_std_dev / math.sqrt(n_samples))
    return float(sample_mean - margin_of_error_val), float(sample_mean + margin_of_error_val)


def calculate_covariance_value(x_data, y_data):
    # 1. Covariance calculation
    n_elements = len(x_data)
    mean_x = np.mean(x_data)
    mean_y = np.mean(y_data)
    covariance_val = sum((x_data[i] - mean_x) * (y_data[i] - mean_y) for i in range(n_elements)) / (n_elements - 1)
    return covariance_val


if __name__ == "__main__":
    # 1. Configuration
    num_servers_config = 10
    service_rate_config = 1 / 8
    arrival_rate_config = 1

    blocking_fractions_list = []
    avg_service_times_list = []
    avg_inter_arrival_times_list = []
    
    # 2. Run trials
    for _ in range(10):
        sim_instance = ErlangBSimulation(num_servers_config, service_rate_config, arrival_rate_config)
        sim_instance.run_simulation()
        blocking_fractions_list.append(sim_instance.get_blocking_fraction())
        avg_service_times_list.append(sim_instance.get_mean_service_time())
        avg_inter_arrival_times_list.append(sim_instance.get_mean_inter_arrival_time())

    # 3. Output basic results
    print(f"--- Erlang B Simulation Results ---")
    analytical_val = calculate_analytical_blocking(num_servers_config, arrival_rate_config, 1/service_rate_config)
    print(f"Analytical Blocking Fraction: {analytical_val:.6f}")
    
    conf_interval = calculate_confidence_interval(blocking_fractions_list)
    print(f"Confidence interval (Crude): {conf_interval}")
    print(f"Variance (Crude): {np.var(blocking_fractions_list, ddof=1):.8f}")
    print()

    # 4. Control Variate: Service Times
    cov_service = calculate_covariance_value(blocking_fractions_list, avg_service_times_list)
    var_service = np.var(avg_service_times_list, ddof=1)
    c_service = -cov_service / var_service
    control_estimates_service = [
        blocking_fractions_list[i] + c_service * (avg_service_times_list[i] - 8)
        for i in range(len(blocking_fractions_list))
    ]

    ci_lower_service, ci_upper_service = calculate_confidence_interval(control_estimates_service)
    print(f"--- Control Variate: Service Times ---")
    print(f"Confidence Interval: [{ci_lower_service:.4f}, {ci_upper_service:.4f}]")
    print(f"Variance: {np.var(control_estimates_service, ddof=1):.8f}")
    print()

    # 5. Control Variate: Inter-arrival Times
    cov_arrivals = calculate_covariance_value(blocking_fractions_list, avg_inter_arrival_times_list)
    var_arrivals = np.var(avg_inter_arrival_times_list, ddof=1)
    c_arrivals = -cov_arrivals / var_arrivals
    control_estimates_arrivals = [
        blocking_fractions_list[i] + c_arrivals * (avg_inter_arrival_times_list[i] - 8)
        for i in range(len(blocking_fractions_list))
    ]

    ci_lower_arrivals, ci_upper_arrivals = calculate_confidence_interval(control_estimates_arrivals)
    print(f"--- Control Variate: Inter-arrival Times ---")
    print(f"Confidence Interval: [{ci_lower_arrivals:.4f}, {ci_upper_arrivals:.4f}]")
    print(f"Variance: {np.var(control_estimates_arrivals, ddof=1):.8f}")
