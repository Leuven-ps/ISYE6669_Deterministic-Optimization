from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def f(x: np.ndarray) -> float:
    """Objective function f(x1, x2) = (2x1 + x2^2 - 4)^2 + (x1^2 + x2 - 8)^2"""
    x1, x2 = x[0], x[1]
    return (2*x1 + x2**2 - 4)**2 + (x1**2 + x2 - 8)**2


def gradient_f(x: np.ndarray) -> np.ndarray:
    """Gradient of f(x1, x2)"""
    x1, x2 = x[0], x[1]
    grad_x1 = 4*x1**3 + 4*x1*x2 - 24*x1 + 4*x2**2 - 16
    grad_x2 = 4*x2**3 + 8*x1*x2 - 14*x2 + 2*x1**2 - 16
    return np.array([grad_x1, grad_x2])


def hessian_f(x: np.ndarray) -> np.ndarray:
    """Hessian matrix of f(x1, x2)"""
    x1, x2 = x[0], x[1]
    h11 = 12*x1**2 + 4*x2 - 24
    h12 = 4*x1 + 8*x2
    h21 = 4*x1 + 8*x2
    h22 = 12*x2**2 + 8*x1 - 14
    return np.array([[h11, h12], [h21, h22]])


def newton_method_with_line_search(
    x0: np.ndarray,
    alpha_bar: float = 1.0,
    rho: float = 0.5,
    c: float = 0.1,
    epsilon: float = 1e-4,
    max_iter: int = 100
) -> Tuple[np.ndarray, list, list]:
    """
    Newton's method with line search

    Args:
        x0: Initial point
        alpha_bar: Initial step size
        rho: Step size reduction factor
        c: Armijo condition parameter
        epsilon: Convergence tolerance
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (final point, list of points, list of step sizes)
    """
    x = x0.copy()
    points = [x.copy()]
    step_sizes = []

    for k in range(max_iter):
        # Compute gradient and Hessian
        grad = gradient_f(x)

        # Check convergence
        if np.linalg.norm(grad) <= epsilon:
            break

        # Compute Newton direction
        hess = hessian_f(x)
        try:
            d = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use gradient descent direction
            d = -grad

        # Line search
        alpha = alpha_bar
        while f(x + alpha * d) > f(x) + c * alpha * np.dot(grad, d):
            alpha = rho * alpha

        # Update point
        x = x + alpha * d
        points.append(x.copy())
        step_sizes.append(alpha)

        print(f"Step {k}: x = [{x[0]:.6f}, {x[1]:.6f}], alpha = {alpha:.6f}")

    return x, points, step_sizes


def plot_function_2d_3d():
    """Plot the function f(x1, x2) in 2D contour and 3D surface"""
    # Create grid for plotting
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Calculate function values
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 6))

    # 2D Contour plot
    ax1 = fig.add_subplot(121)
    contour = ax1.contour(X1, X2, Z, levels=20)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('2D Contour Plot of f(x1, x2)')
    ax1.grid(True, alpha=0.3)

    # 3D Surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x1, x2)')
    ax2.set_title('3D Surface Plot of f(x1, x2)')

    # Add colorbar
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()


def plot_newton_convergence(points_1: list, points_2: list):
    """Plot Newton's method convergence paths"""
    # Create grid for plotting
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Calculate function values
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Starting point (-3, -3)
    contour1 = ax1.contour(X1, X2, Z, levels=20, alpha=0.6)
    ax1.clabel(contour1, inline=True, fontsize=8)

    # Plot convergence path
    points_array_1 = np.array(points_1)
    ax1.plot(points_array_1[:, 0], points_array_1[:, 1], 'ro-',
             markersize=6, linewidth=2, label='Convergence path')
    ax1.plot(points_array_1[0, 0], points_array_1[0, 1], 'go',
             markersize=10, label='Start: (-3, -3)')
    ax1.plot(points_array_1[-1, 0], points_array_1[-1, 1], 'r*',
             markersize=15, label='Final point')

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Newton Method: Starting from (-3, -3)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Starting point (-2, -2)
    contour2 = ax2.contour(X1, X2, Z, levels=20, alpha=0.6)
    ax2.clabel(contour2, inline=True, fontsize=8)

    # Plot convergence path
    points_array_2 = np.array(points_2)
    ax2.plot(points_array_2[:, 0], points_array_2[:, 1], 'bo-',
             markersize=6, linewidth=2, label='Convergence path')
    ax2.plot(points_array_2[0, 0], points_array_2[0, 1], 'go',
             markersize=10, label='Start: (-2, -2)')
    ax2.plot(points_array_2[-1, 0], points_array_2[-1, 1], 'b*',
             markersize=15, label='Final point')

    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Newton Method: Starting from (-2, -2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run Newton's method for different starting points"""
    print("=== Newton's Method with Line Search ===")
    print("Function: f(x1, x2) = (2x1 + x2^2 - 4)^2 + (x1^2 + x2 - 8)^2")
    print()

    # Plot the function first
    print("Plotting the function...")
    plot_function_2d_3d()

    # Starting point 1: (-3, -3)
    print("Starting point 1: x^0 = (-3, -3)")
    print("-" * 50)
    x0_1 = np.array([-3.0, -3.0])
    x_final_1, points_1, step_sizes_1 = newton_method_with_line_search(x0_1)
    print(f"Final point: [{x_final_1[0]:.6f}, {x_final_1[1]:.6f}]")
    print(f"Final function value: {f(x_final_1):.6f}")
    print(f"Number of iterations: {len(step_sizes_1)}")
    print()

    # Starting point 2: (-2, -2)
    print("Starting point 2: x^0 = (-2, -2)")
    print("-" * 50)
    x0_2 = np.array([-2.0, -2.0])
    x_final_2, points_2, step_sizes_2 = newton_method_with_line_search(x0_2)
    print(f"Final point: [{x_final_2[0]:.6f}, {x_final_2[1]:.6f}]")
    print(f"Final function value: {f(x_final_2):.6f}")
    print(f"Number of iterations: {len(step_sizes_2)}")
    print()

    # Comparison
    print("=== Comparison ===")
    print(f"Starting point (-3, -3): {len(step_sizes_1)} iterations, "
          f"final value = {f(x_final_1):.6f}")
    print(f"Starting point (-2, -2): {len(step_sizes_2)} iterations, "
          f"final value = {f(x_final_2):.6f}")

    if len(step_sizes_1) < len(step_sizes_2):
        print("Starting point (-3, -3) is better (fewer iterations)")
    elif len(step_sizes_2) < len(step_sizes_1):
        print("Starting point (-2, -2) is better (fewer iterations)")
    else:
        print("Both starting points perform equally well")

    # Plot convergence paths
    print("\nPlotting convergence paths...")
    plot_newton_convergence(points_1, points_2)


if __name__ == "__main__":
    main()
