import numpy as np
import random as rand


def read_csv(filename):
    return np.loadtxt(filename, delimiter=",")


def calculate_cost(cost_matrix, path):
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += cost_matrix[path[i]][path[i + 1]]
    total_cost += cost_matrix[path[-1]][path[0]]  # Return to the starting point
    return total_cost


def new_proposal(path):
    new_path = path.copy()
    i, j = np.random.choice(len(path), 2, replace=False)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def acceptance_probability(cost_matrix, current_path, new_path, temperature):
    current_cost = calculate_cost(cost_matrix, current_path)
    new_cost = calculate_cost(cost_matrix, new_path)
    if new_cost < current_cost:
        return 1.0
    else:
        return np.exp((current_cost - new_cost) / temperature)


def simulated_annealing(cost_matrix, initial_path, iterations):
    current_path = initial_path
    current_cost = calculate_cost(cost_matrix, current_path)
    best_path = current_path
    best_cost = current_cost

    for iteration in range(iterations):
        new_path = new_proposal(current_path)
        temperature = 1 / np.sqrt(iteration + 1)
        ap = acceptance_probability(cost_matrix, current_path, new_path, temperature)

        if np.random.rand() < ap:
            current_path = new_path
            current_cost = calculate_cost(cost_matrix, current_path)

            if current_cost < best_cost:
                best_path = current_path
                best_cost = current_cost

        # Cool down the temperature

    return best_path, best_cost


if __name__ == "__main__":
    cost_matrix = read_csv("/Users/sushantyadav/Downloads/cost.csv")
    # print(cost_matrix)
    best_path, best_cost = simulated_annealing(
        cost_matrix,
        initial_path=rand.sample(list(range(len(cost_matrix))), len(cost_matrix)),
        iterations=10,
    )
    # print(f"Best path: {best_path}, Best cost: {best_cost}")

    for i in range(1000):
        path, cost = simulated_annealing(
            cost_matrix,
            initial_path=rand.sample(list(range(len(cost_matrix))), len(cost_matrix)),
            iterations=1000,
        )

        if cost < best_cost:
            best_path, best_cost = path, cost

    print(f"Best path: {best_path}, Best cost: {best_cost}")
