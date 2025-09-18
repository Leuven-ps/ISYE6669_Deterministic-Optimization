import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def f(x: float) -> float:
    """Objective function f(x) = -x + 2^x"""
    return -x + 2**x


def f_prime(x: float) -> float:
    """First derivative f'(x) = -1 + 2^x * ln(2)"""
    return -1 + 2**x * np.log(2)


def f_double_prime(x: float) -> float:
    """Second derivative f''(x) = 2^x * (ln(2))^2"""
    return 2**x * (np.log(2))**2


def newton_method(x0: float, tol: float = 1e-5, max_iter: int = 10) -> tuple:
    """
    Newton's method for minimizing f(x) = -x + 2^x
    
    Args:
        x0: Initial point
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        
    Returns:
        tuple: (solution, iterations_data)
    """
    x = x0
    iterations = []
    
    for k in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        fppx = f_double_prime(x)
        
        iterations.append({
            'k': k,
            'x': x,
            'f(x)': fx,
            "f'(x)": fpx
        })
        
        # Check convergence
        if abs(fpx) < tol:
            break
            
        # Newton update
        x = x - fpx / fppx
    
    return x, iterations


def plot_newton_geometry():
    """Plot the geometric interpretation of Newton's method"""
    x = np.linspace(-1, 5, 1000)
    y = f_prime(x)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'b-', linewidth=2, label="y = f'(x)")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='y = 0')
    
    # Newton iterations
    x0 = 0
    x_vals = [x0]
    y_vals = [f_prime(x0)]
    
    for i in range(3):
        x_curr = x_vals[-1]
        y_curr = y_vals[-1]
        
        # Tangent line at current point
        slope = f_double_prime(x_curr)
        x_tangent = np.linspace(x_curr - 1, x_curr + 2, 100)
        y_tangent = slope * (x_tangent - x_curr) + y_curr
        
        plt.plot(x_tangent, y_tangent, 'r--', alpha=0.7, 
                label=f'Tangent at x^{i}' if i < 3 else '')
        
        # Find zero of tangent line
        x_next = x_curr - y_curr / slope
        x_vals.append(x_next)
        y_vals.append(f_prime(x_next))
        
        # Plot points
        plt.plot(x_curr, y_curr, 'ro', markersize=8)
        plt.plot(x_next, 0, 'go', markersize=8)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Geometric Interpretation of Newton\'s Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 5)
    plt.ylim(-2, 3)
    plt.show()


def problem_2_1():
    """Solve problem 2.1: Minimize the given function"""
    def objective(x):
        x1, x2 = x[0], x[1]
        return (1 - x2 + x1*x2)**2 + (2 - x2 + x1*x2**2)**2 + (3 - x2 + x1*x2**3)**2
    
    # Optimization
    x0 = [0, 0]
    bounds = [(-5, 5), (-5, 5)]
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    print("Problem 2.1 Results:")
    print(f"Optimal solution: x1 = {result.x[0]:.6f}, x2 = {result.x[1]:.6f}")
    print(f"Optimal value: {result.fun:.6f}")
    print(f"Success: {result.success}")
    
    # Plotting
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective([X1, X2])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Contour plot
    contour = ax1.contour(X1, X2, Z, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(result.x[0], result.x[1], 'r*', markersize=15, label='Optimal point')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('2D Contour Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax2.scatter([result.x[0]], [result.x[1]], [result.fun], 
               color='red', s=100, label='Optimal point')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x1, x2)')
    ax2.set_title('3D Plot')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def problem_2_2():
    """Solve problem 2.2: Find multiple local minima"""
    def objective(x):
        x1, x2 = x[0], x[1]
        return (2*x1 + x2**2 - 4)**2 + (x1**2 + x2 - 8)**2
    
    # Multiple starting points
    starting_points = [
        [0, 0], [2, 2], [-2, 2], [2, -2], [-2, -2],
        [3, 3], [-3, 3], [3, -3], [-3, -3], [1, 4]
    ]
    
    bounds = [(-5, 5), (-5, 5)]
    local_minima = []
    
    print("Problem 2.2 Results:")
    for i, x0 in enumerate(starting_points):
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        local_minima.append((result.x, result.fun, result.success))
        print(f"Starting point {x0}: solution={result.x}, value={result.fun:.6f}, success={result.success}")
    
    # Remove duplicates to identify unique local minima
    unique_minima = []
    for x, val, success in local_minima:
        if success:
            is_unique = True
            for ux, uval in unique_minima:
                if np.linalg.norm(np.array(x) - np.array(ux)) < 1e-3:
                    is_unique = False
                    break
            if is_unique:
                unique_minima.append((x, val))
    
    print(f"\nNumber of local minima found: {len(unique_minima)}")
    for i, (x, val) in enumerate(unique_minima):
        print(f"Local minimum {i+1}: x1={x[0]:.6f}, x2={x[1]:.6f}, f(x)={val:.6f}")
    
    # Plotting
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective([X1, X2])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Contour plot
    contour = ax1.contour(X1, X2, Z, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    for i, (x, val) in enumerate(unique_minima):
        ax1.plot(x[0], x[1], 'r*', markersize=15, label=f'Local min {i+1}' if i == 0 else '')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('2D Contour Plot with Local Minima')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    for i, (x, val) in enumerate(unique_minima):
        ax2.scatter([x[0]], [x[1]], [val], 
                   color='red', s=100, label=f'Local min {i+1}' if i == 0 else '')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x1, x2)')
    ax2.set_title('3D Plot with Local Minima')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Problem 1.1: Newton's method
    print("Problem 1.1: Newton's Method")
    print("=" * 40)
    solution, iterations = newton_method(x0=0, tol=1e-5, max_iter=10)
    
    print("Iteration results:")
    for iter_data in iterations:
        f_prime_key = "f'(x)"
        print(f"k={iter_data['k']}: x^k={iter_data['x']:.6f}, "
              f"f(x^k)={iter_data['f(x)']:.6f}, f'(x^k)={iter_data[f_prime_key]:.6f}")
    
    print(f"\nFinal solution: x* = {solution:.6f}")
    print(f"Final f'(x*) = {f_prime(solution):.6f}")
    
    # Plot geometric interpretation
    print("\nPlotting geometric interpretation...")
    plot_newton_geometry()
    
    # Problem 2.1
    print("\n" + "=" * 50)
    problem_2_1()
    
    # Problem 2.2
    print("\n" + "=" * 50)
    problem_2_2()
