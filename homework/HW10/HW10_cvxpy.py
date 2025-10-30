"""
ISyE6669 Homework 10: Cutting Stock Problem with Column Generation
RMP (Restricted Master Problem) implementation using CVXPy
"""

import cvxpy as cp
import numpy as np


def solve_rmp(patterns: list, demands: list) -> dict:
    """
    Solve the Restricted Master Problem (RMP)
    
    Args:
        patterns: List of pattern matrices [A1, A2, A3, ...]
        demands: Demand vector [b1, b2, b3]
    
    Returns:
        Dictionary containing optimal solution, objective value, and model
    """
    # Create variables
    n_patterns = len(patterns)
    x = cp.Variable(n_patterns, nonneg=True)
    
    # Objective function: minimize sum(x_j)
    objective = cp.Minimize(cp.sum(x))
    
    # Constraints: sum(A_j * x_j) = b
    constraints = []
    for i in range(len(demands)):
        constraint_expr = 0
        for j in range(n_patterns):
            constraint_expr += patterns[j][i] * x[j]
        constraints.append(constraint_expr == demands[i])
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    if problem.status != cp.OPTIMAL:
        raise Exception(f"Optimization failed. Status: {problem.status}")
    
    # Get optimal solution
    optimal_x = x.value
    
    return {
        "optimal_x": optimal_x,
        "objective_value": problem.value,
        "problem": problem
    }


def find_basis_and_dual(patterns: list, demands: list, optimal_x: list) -> dict:
    """
    Manually calculate basis and dual solution
    
    Args:
        patterns: List of pattern matrices
        demands: Demand vector
        optimal_x: Optimal solution
    
    Returns:
        Dictionary containing basis, basis inverse, and dual solution
    """
    # Get indices of non-zero variables
    basic_vars = [i for i, val in enumerate(optimal_x) if abs(val) > 1e-6]
    
    if len(basic_vars) != len(demands):
        raise Exception("Number of basic variables does not match number of constraints")
    
    # Construct basis matrix B
    B = np.array([[patterns[j][i] for j in basic_vars] for i in range(len(demands))])
    
    # Calculate basis inverse
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise Exception("Basis matrix is singular")
    
    # Calculate dual solution: y^T = c_B^T * B^(-1)
    # Here c_B = [1, 1, ..., 1] (objective function coefficients)
    c_B = np.ones(len(basic_vars))
    dual_solution = c_B.T @ B_inv
    
    return {
        "basic_vars": basic_vars,
        "basis_matrix": B,
        "basis_inverse": B_inv,
        "dual_solution": dual_solution
    }


def solve_pricing_problem(dual_solution: list, widths: list, max_width: int) -> dict:
    """
    Solve the pricing problem (knapsack problem)
    
    Args:
        dual_solution: Dual solution [y1, y2, y3]
        widths: Small roll widths [w1, w2, w3]
        max_width: Large roll width W
    
    Returns:
        Dictionary containing optimal solution and new pattern
    """
    # Create variables: a1, a2, a3 (number of each width)
    a = cp.Variable(3, integer=True, nonneg=True)
    
    # Objective function: maximize sum(y_i * a_i)
    # If optimal objective value > 1, then improvement is possible
    # If optimal objective value <= 1, then no improvement possible
    objective = cp.Maximize(cp.sum([dual_solution[i] * a[i] for i in range(3)]))
    
    # Constraint: sum(w_i * a_i) <= W
    constraints = [cp.sum([widths[i] * a[i] for i in range(3)]) <= max_width]
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise Exception(f"Pricing problem optimization failed. Status: {problem.status}")
    
    # Get optimal solution
    optimal_a = a.value
    objective_value = problem.value
    
    return {
        "optimal_a": optimal_a,
        "objective_value": objective_value,
        "new_pattern": [int(optimal_a[i]) for i in range(3)]
    }


def column_generation_iteration(patterns: list, demands: list, widths: list, max_width: int) -> dict:
    """
    Execute one iteration of column generation
    
    Args:
        patterns: Current pattern list
        demands: Demand vector
        widths: Small roll widths
        max_width: Large roll width
    
    Returns:
        Dictionary containing iteration results
    """
    print("=== Executing Column Generation Iteration ===")
    
    # Solve RMP
    rmp_result = solve_rmp(patterns, demands)
    print(f"RMP optimal solution: {rmp_result['optimal_x']}")
    print(f"RMP optimal objective value: {rmp_result['objective_value']}")
    
    # Calculate basis and dual solution
    basis_result = find_basis_and_dual(patterns, demands, rmp_result['optimal_x'])
    print(f"Dual solution: {basis_result['dual_solution']}")
    
    # Solve pricing problem
    pricing_result = solve_pricing_problem(
        basis_result['dual_solution'], widths, max_width
    )
    print(f"New pattern: {pricing_result['new_pattern']}")
    print(f"Objective value: {pricing_result['objective_value']:.6f}")
    
    return {
        "rmp_result": rmp_result,
        "basis_result": basis_result,
        "pricing_result": pricing_result
    }


def full_column_generation(demands: list, widths: list, max_width: int, initial_patterns: list) -> dict:
    """
    Execute complete column generation algorithm
    
    Args:
        demands: Demand vector
        widths: Small roll widths
        max_width: Large roll width
        initial_patterns: Initial patterns
    
    Returns:
        Dictionary containing final results
    """
    patterns = initial_patterns.copy()
    iteration = 0
    max_iterations = 10  # Safety limit for maximum iterations
    
    print("=== Starting Complete Column Generation Algorithm ===")
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Solve RMP
        rmp_result = solve_rmp(patterns, demands)
        print(f"RMP optimal solution: {rmp_result['optimal_x']}")
        print(f"RMP optimal objective value: {rmp_result['objective_value']}")
        
        # Calculate basis and dual solution
        basis_result = find_basis_and_dual(patterns, demands, rmp_result['optimal_x'])
        print(f"Dual solution: {basis_result['dual_solution']}")
        
        # Solve pricing problem
        pricing_result = solve_pricing_problem(
            basis_result['dual_solution'], widths, max_width
        )
        print(f"New pattern: {pricing_result['new_pattern']}")
        print(f"Objective value: {pricing_result['objective_value']:.6f}")
        
        # Check termination condition: objective value <= 1 means no improvement
        if pricing_result['objective_value'] <= 1.0 + 1e-6:
            print("Objective value <= 1, terminating column generation")
            break
        else:
            print("Objective value > 1, continuing column generation")
            # Add new pattern
            patterns.append(pricing_result['new_pattern'])
            print(f"Updated pattern list: {patterns}")
    
    if iteration >= max_iterations:
        print(f"Warning: Reached maximum iterations {max_iterations}")
    
    # Return final results
    final_rmp = solve_rmp(patterns, demands)
    final_basis = find_basis_and_dual(patterns, demands, final_rmp['optimal_x'])
    
    return {
        "final_patterns": patterns,
        "final_rmp": final_rmp,
        "final_basis": final_basis,
        "iterations": iteration
    }


def main():
    """Main function"""
    print("=== ISyE6669 Homework 10: Cutting Stock Problem ===\n")
    
    # Problem data
    demands = [25, 15, 10]  # b1, b2, b3
    widths = [20, 35, 45]   # w1, w2, w3
    max_width = 100         # W
    initial_patterns = [
        [5, 0, 0],  # A1
        [0, 2, 0],  # A2
        [0, 0, 2]   # A3
    ]
    
    print("Problem data:")
    print(f"Demands: {demands}")
    print(f"Small roll widths: {widths}")
    print(f"Large roll width: {max_width}")
    print(f"Initial patterns: {initial_patterns}")
    print()
    
    # Execute complete column generation
    result = full_column_generation(demands, widths, max_width, initial_patterns)
    
    print("\n=== Final Results ===")
    print(f"Final number of patterns: {len(result['final_patterns'])}")
    print(f"Final patterns:")
    for i, pattern in enumerate(result['final_patterns']):
        print(f"  Pattern {i+1}: {pattern}")
    
    print(f"\nFinal RMP optimal solution: {result['final_rmp']['optimal_x']}")
    print(f"Final RMP optimal objective value: {result['final_rmp']['objective_value']}")
    
    print(f"\nFinal basic variables: {result['final_basis']['basic_vars']}")
    print(f"Final basis matrix B:")
    print(result['final_basis']['basis_matrix'])
    print(f"\nFinal basis inverse B^(-1):")
    print(result['final_basis']['basis_inverse'])
    print(f"\nFinal dual solution y^T: {result['final_basis']['dual_solution']}")
    
    print(f"\nTotal iterations: {result['iterations']}")
    
    return result


if __name__ == "__main__":
    main()

