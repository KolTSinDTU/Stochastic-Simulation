import math
import random as rand
import matplotlib.pyplot as plt

def calculate_distance(point1, point2):
    # 1. Calculate Euclidean distance between two points
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def calculate_total_cost(points, order):
    # 1. Sum up distances between consecutive points in the order
    total_dist = 0
    num_points = len(order)
    for i in range(num_points):
        total_dist += calculate_distance(points[order[i]], points[order[(i + 1) % num_points]])
    return total_dist

def generate_neighbor_route(order):
    # 1. Create a copy and swap two random cities
    new_order = list(order)
    idx1, idx2 = rand.sample(range(len(order)), 2)
    new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]
    return new_order

def simulated_annealing_tsp(points, max_iterations=50000):
    # 1. Initialize route and costs
    current_order = list(range(len(points)))
    best_order = list(current_order)
    min_cost = calculate_total_cost(points, current_order)
    current_cost = min_cost
    
    # 2. Main simulated annealing loop
    for k in range(max_iterations):
        # 3. Update temperature based on iteration
        temperature = 1 / math.sqrt(1 + k)
        
        # 4. Propose neighbor and calculate its cost
        new_order = generate_neighbor_route(current_order)
        new_cost = calculate_total_cost(points, new_order)
        
        # 5. Acceptance criteria (Metropolis-Hastings)
        if new_cost < current_cost or rand.random() < math.exp(-(new_cost - current_cost) / temperature):
            current_order = new_order
            current_cost = new_cost
            # 6. Track the best solution found so far
            if current_cost < min_cost:
                min_cost = current_cost
                best_order = list(current_order)
                
    return best_order

if __name__ == "__main__":
    # 1. Generate points on a unit circle
    num_cities = 20
    city_points = [(math.cos(2 * math.pi * i / num_cities), math.sin(2 * math.pi * i / num_cities)) for i in range(num_cities)]
    rand.shuffle(city_points)

    # 2. Run TSP optimization
    best_route_indices = simulated_annealing_tsp(city_points)
    
    # 3. Prepare data for plotting the route
    x_coords = [city_points[i][0] for i in best_route_indices] + [city_points[best_route_indices[0]][0]]
    y_coords = [city_points[i][1] for i in best_route_indices] + [city_points[best_route_indices[0]][1]]

    # 4. Visualize the optimized route
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_coords, y_coords, '-o', color="#2c3e50", markersize=8, markerfacecolor="#e74c3c")
    ax.set_aspect('equal')
    ax.set_title(f"TSP Solution via Simulated Annealing")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Output summary results
    print(f"--- TSP Optimization Results ---")
    print(f"Final Optimized Cost: {calculate_total_cost(city_points, best_route_indices):.4f}")
    
    plt.tight_layout()
    plt.show()
