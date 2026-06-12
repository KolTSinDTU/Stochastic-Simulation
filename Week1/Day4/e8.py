import numpy as np
import math
import random as rand
from matplotlib import pyplot as plt

def sample_from_g(lambda_val, num_samples):
    # 1. This generates samples from the distribution g(x) = lambda * e^(-lambda * x)
    return [-math.log(rand.random())/ lambda_val for _ in range(num_samples)]


def calculate_numerator_hf(samples):
    # 1. This evaluates the actual function: h(x) = e^x if 0 <= x <= 1, else 0 for all samples
    return [math.exp(x) if 0 <=x <= 1 else 0 for x in samples]

def calculate_denominator_g(samples, lambda_val):
    # 1. This evaluates the actual function: g(x) = lambda * e^(-lambda * x) for all samples
    return [lambda_val * math.exp(-lambda_val * x) for x in samples]

def simulate_variance_for_lambda(lambda_val, num_samples):
    
    samples = sample_from_g(lambda_val, num_samples)
    numerator = calculate_numerator_hf(samples)
    denominator = calculate_denominator_g(samples, lambda_val)
    estimator_values = [num / denom for num, denom in zip(numerator, denominator)]
   
    variance = np.var(estimator_values, ddof=1)
    return variance

def find_optimal_lambda(possible_lambdas, num_samples=10000):
    # 1. Create variables to keep track of the lowest variance and the lambda that caused it 
    #  Return the optimal lambda and its variance
    optimal_lambda = None
    lowest_variance = float('inf')
    for lambda_val in possible_lambdas:
        variance = simulate_variance_for_lambda(lambda_val, num_samples)
        if variance < lowest_variance:
            lowest_variance = variance
            optimal_lambda = lambda_val
    return optimal_lambda, lowest_variance

def find_optimal_lambda_for_plot(possible_lambdas, num_samples=10000):
    variances = [] # Store all variances here
    
    for lambda_val in possible_lambdas:
        variance = simulate_variance_for_lambda(lambda_val, num_samples)
        variances.append(variance)
        
    return possible_lambdas, variances

def plot_variance_curve(lambdas, variances):
    # Create a nice, wide canvas
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot the main curve
    ax.plot(lambdas, variances, color='#2c3e50', linewidth=2.5, label='Simulated Variance')
    
    # 2. Find the exact minimum point mathematically
    min_index = np.argmin(variances)
    optimal_lambda = lambdas[min_index]
    lowest_variance = variances[min_index]
    
    # 3. Drop a big, highly visible red dot right on the minimum
    ax.scatter(optimal_lambda, lowest_variance, color='#e74c3c', s=120, zorder=5, 
               label=f'Optimal $\lambda \\approx$ {optimal_lambda:.2f}')
    
    # 4. Draw an arrow pointing to the minimum with the exact numbers
    # We use a little math (optimal_lambda + 0.1) to offset the text so it doesn't cover the line
    ax.annotate(f'Minimum\n({optimal_lambda:.2f}, {lowest_variance:.2f})',
                xy=(optimal_lambda, lowest_variance),
                xytext=(optimal_lambda + 0.15, lowest_variance + 1.0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold')

    # 5. Add professional labels, titles, and grid lines
    ax.set_title('Importance Sampling: Estimator Variance vs. Tuning Parameter $\lambda$', fontsize=14, pad=15)
    ax.set_xlabel('$\lambda$ (Rate Parameter)', fontsize=12)
    ax.set_ylabel('Sample Variance', fontsize=12)
    
    # Add a subtle grid to make reading values easier
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show the legend
    ax.legend(fontsize=12, loc='upper right')
    
    # Clean up the layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    possible_lambdas = np.linspace(0.1, 3.0, 100)  # Example range of lambda values
    optimal_lambda, lowest_variance = find_optimal_lambda(possible_lambdas)
    print(f"Optimal Lambda: {optimal_lambda}")
    print(f"Lowest Variance: {lowest_variance}")
    lambdas, variances = find_optimal_lambda_for_plot(possible_lambdas)
    plot_variance_curve(lambdas, variances) 
