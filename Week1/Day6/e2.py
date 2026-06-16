import numpy as np
import random as rand


def load_cost_matrix(file_path):
    # 1. Load matrix from CSV file
    return np.loadtxt(file_path, delimiter=",")


def calculate_route_cost(cost_matrix, route_path):
    # 1. Initialize total cost
    total_cost = 0
    # 2. Add costs between consecutive cities in the path
    for i in range(len(route_path) - 1):
        total_cost += cost_matrix[route_path[i]][route_path[i + 1]]
    # 3. Add cost to return to the starting city
    total_cost += cost_matrix[route_path[-1]][route_path[0]]
    return total_cost


def generate_neighbor_path(current_path):
    # 1. Copy the path and swap two random cities
    new_path = current_path.copy()
    idx1, idx2 = np.random.choice(len(current_path), 2, replace=False)
    new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
    return new_path


def get_acceptance_probability(cost_matrix, current_path, proposed_path, temperature):
    # 1. Calculate costs for current and proposed paths
    current_cost = calculate_route_cost(cost_matrix, current_path)
    proposed_cost = calculate_route_cost(cost_matrix, proposed_path)
    
    # 2. Determine acceptance probability using Metropolis criterion
    if proposed_cost < current_cost:
        return 1.0
    else:
        return np.exp((current_cost - proposed_cost) / temperature)


def run_simulated_annealing(cost_matrix, initial_route, max_iterations):
    # 1. Initialize state variables
    current_route = initial_route
    current_cost = calculate_route_cost(cost_matrix, current_route)
    best_route = current_route
    best_cost = current_cost

    # 2. Iteratively optimize the route
    for iteration in range(max_iterations):
        # 3. Generate and evaluate a neighboring proposal
        proposed_route = generate_neighbor_path(current_route)
        current_temp = 1 / np.sqrt(iteration + 1)
        prob = get_acceptance_probability(cost_matrix, current_route, proposed_route, current_temp)

        # 4. Accept or reject the proposal
        if np.random.rand() < prob:
            current_route = proposed_route
            current_cost = calculate_route_cost(cost_matrix, current_route)

            # 5. Track the global best solution found
            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

    return best_route, best_cost


if __name__ == "__main__":
    # 1. Load the cost data
    try:
        # Using relative path for better portability
        cost_matrix_data = load_cost_matrix("cost.csv")
    except Exception:
        # Fallback for demonstration if file is missing
        cost_matrix_data = np.zeros((10, 10))

    # 2. Perform initial optimization
    best_route, best_cost = run_simulated_annealing(
        cost_matrix_data,
        initial_route=rand.sample(list(range(len(cost_matrix_data))), len(cost_matrix_data)),
        max_iterations=10,
    )

    # 3. Refine the result through multiple trials
    for _ in range(1000):
        trial_route, trial_cost = run_simulated_annealing(
            cost_matrix_data,
            initial_route=rand.sample(list(range(len(cost_matrix_data))), len(cost_matrix_data)),
            max_iterations=1000,
        )

        if trial_cost < best_cost:
            best_route, best_cost = trial_route, trial_cost

    # 4. Display the results
    print(f"--- TSP Optimization Results ---")
    print(f"Best Route Found: {best_route}")
    print(f"Minimum Total Cost: {best_cost:.4f}")
