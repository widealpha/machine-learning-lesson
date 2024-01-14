import math
import random


def simulated_annealing(initial_solution, objective_function, generate_neighbor, initial_temperature, cooling_rate,
                        stopping_condition):
    current_solution = initial_solution
    current_temperature = initial_temperature
    k = 0

    while not stopping_condition(k, current_temperature):
        neighbor_solution = generate_neighbor(current_solution)
        delta_f = objective_function(neighbor_solution) - objective_function(current_solution)

        if delta_f <= 0 or random.uniform(0, 1) < math.exp(-delta_f / current_temperature):
            current_solution = neighbor_solution

        current_temperature = cooling_rate(current_temperature)
        k += 1

    return current_solution


# Example Usage:
# Define your own objective function, initial solution, and neighbor generation function
def objective_function(x):
    # Your objective function to minimize
    return sum(x)


def generate_neighbor(current_solution):
    # Your method to generate a neighboring solution
    neighbor_solution = current_solution.copy()
    index = random.randint(0, len(neighbor_solution) - 1)
    neighbor_solution[index] = 1 - neighbor_solution[index]  # Flip 0 to 1 or 1 to 0
    return neighbor_solution


def cooling_rate(current_temperature):
    # Your cooling rate function
    return 0.95 * current_temperature  # Example cooling rate


def stopping_condition(k, current_temperature):
    # Your stopping condition, e.g., a maximum number of iterations or minimum temperature
    return k < 1000 and current_temperature > 0.01


# Example usage:
initial_solution = [0, 1, 0, 1, 1]
result = simulated_annealing(initial_solution, objective_function, generate_neighbor, initial_temperature=100,
                             cooling_rate=cooling_rate, stopping_condition=stopping_condition)

print("Best solution:", result)
print("Objective function value:", objective_function(result))
