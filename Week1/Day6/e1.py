import math
import random
import matplotlib.pyplot as plt

def calculate_euclidean_distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def calculate_total_route_cost(points, route):
    total_distance = 0
    num_points = len(route)
    for i in range(num_points):
        p1 = points[route[i]]
        p2 = points[route[(i + 1) % num_points]]
        total_distance += calculate_euclidean_distance(p1, p2)
    return total_distance

def generate_proposal(route):
    new_route = list(route)
    idx1, idx2 = random.sample(range(len(route)), 2)
    new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
    return new_route

def simulated_annealing(points, max_iterations=50000):
    num_points = len(points)
    
    # Initialize with a completely random route
    current_route = list(range(num_points))
    random.shuffle(current_route)
    initial_route = list(current_route) # Saved to plot the "before" state
    
    best_route = list(current_route)
    current_cost = calculate_total_route_cost(points, current_route)
    best_cost = current_cost
    
    for k in range(max_iterations):
        # Apply the cooling scheme from the prompt: Tk = 1 / sqrt(1 + k)
        temperature = 1.0 / math.sqrt(1 + k)
        
        proposed_route = generate_proposal(current_route)
        proposed_cost = calculate_total_route_cost(points, proposed_route)
        
        # Accept the proposal if it's better, OR with a probability based on temperature
        if proposed_cost < current_cost or random.random() < math.exp(-(proposed_cost - current_cost) / temperature):
            current_route = proposed_route
            current_cost = proposed_cost
            
            # Keep track of the absolute best route found
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = list(current_route)
                
    return initial_route, best_route

n_points = 20

circle_points = [
    (math.cos(2 * math.pi * i / n_points), math.sin(2 * math.pi * i / n_points)) 
    for i in range(n_points)
]

initial_path, optimized_path = simulated_annealing(circle_points)

def get_plot_coordinates(points, route):
    """Extracts X and Y coordinates in order, appending the start point to close the loop."""
    x = [points[i][0] for i in route] + [points[route[0]][0]]
    y = [points[i][1] for i in route] + [points[route[0]][1]]
    return x, y

init_x, init_y = get_plot_coordinates(circle_points, initial_path)
opt_x, opt_y = get_plot_coordinates(circle_points, optimized_path)

plt.figure(figsize=(10, 8))

# Plot the chaotic starting route
plt.plot(init_x, init_y, linestyle='--', color='gray', alpha=0.5, label='Initial Random Route')

# Plot the final optimized route
plt.plot(opt_x, opt_y, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8, label='Optimized Route')

plt.title("Simulated Annealing TSP: Circle Sanity Check")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend(loc="upper right", shadow=True)
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()