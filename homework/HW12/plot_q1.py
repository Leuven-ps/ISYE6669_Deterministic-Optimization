import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (required for 3D projection)
from scipy.linalg import norm


def ensure_output_dir(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def setup_3d_ax(title: str, lim: Tuple[float, float] = (-3.0, 6.0)) -> plt.Axes:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*lim)
    return ax


def draw_vector(ax: plt.Axes, vec: Tuple[float, float, float], color: str = "crimson") -> None:
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = vec
    ax.quiver(x0, y0, z0, x1, y1, z1, color=color, arrow_length_ratio=0.1, linewidth=2.0)
    ax.scatter([x1], [y1], [z1], color=color, s=50)
    ax.text(x1, y1, z1, f"  x - y = {vec}\n  ||x-y|| = {norm(vec):.3f}", color=color)


def plot_r3_plus(ax: plt.Axes, L: float = 6.0, alpha: float = 0.15) -> None:
    # Draw R^3_+ = {(x, y, z): x >= 0, y >= 0, z >= 0}
    # The three boundary planes x=0, y=0, z=0 all intersect at the origin (0,0,0)
    g = np.linspace(0.0, L, 30)
    
    # Plane x = 0 (y-z plane) - boundary of R^3_+
    # This plane passes through origin and extends in positive y and z directions
    Y, Z = np.meshgrid(g, g)
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, color="tab:blue", alpha=alpha * 1.5, rstride=1, cstride=1, linewidth=0.5)

    # Plane y = 0 (x-z plane) - boundary of R^3_+
    # This plane passes through origin and extends in positive x and z directions
    X, Z = np.meshgrid(g, g)
    Y = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, color="tab:blue", alpha=alpha * 1.5, rstride=1, cstride=1, linewidth=0.5)

    # Plane z = 0 (x-y plane) - boundary of R^3_+
    # This plane passes through origin and extends in positive x and y directions
    X, Y = np.meshgrid(g, g)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, color="tab:blue", alpha=alpha * 1.5, rstride=1, cstride=1, linewidth=0.5)

    # Explicitly mark the origin (0, 0, 0) where all three boundary planes intersect
    ax.scatter([0.0], [0.0], [0.0], color="black", s=100, zorder=10, label="Origin (0,0,0)")
    
    # Draw coordinate axes starting from origin to emphasize the region
    # Positive x-axis (from origin)
    ax.plot([0.0, L], [0.0, 0.0], [0.0, 0.0], color="tab:blue", linewidth=2.5, alpha=0.8)
    # Positive y-axis (from origin)
    ax.plot([0.0, 0.0], [0.0, L], [0.0, 0.0], color="tab:blue", linewidth=2.5, alpha=0.8)
    # Positive z-axis (from origin)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, L], color="tab:blue", linewidth=2.5, alpha=0.8)
    
    # Draw a translucent cube [0, L]^3 to show the region extent
    # This helps visualize that R^3_+ extends infinitely in the positive directions
    # Top face (z = L)
    X, Y = np.meshgrid(g, g)
    Z = np.full_like(X, L)
    ax.plot_surface(X, Y, Z, color="tab:blue", alpha=alpha * 0.3, rstride=5, cstride=5, linewidth=0)
    
    # Back face (y = L)
    X, Z = np.meshgrid(g, g)
    Y = np.full_like(X, L)
    ax.plot_surface(X, Y, Z, color="tab:blue", alpha=alpha * 0.3, rstride=5, cstride=5, linewidth=0)
    
    # Right face (x = L)
    Y, Z = np.meshgrid(g, g)
    X = np.full_like(Y, L)
    ax.plot_surface(X, Y, Z, color="tab:blue", alpha=alpha * 0.3, rstride=5, cstride=5, linewidth=0)

    ax.text(2.0, 2.0, 2.0, r"$\mathbb{R}^3_+$: $x, y, z \geq 0$", color="tab:blue", fontsize=10, weight="bold")


def plot_soc_cone(ax: plt.Axes, R: float = 6.0, alpha: float = 0.15) -> None:
    # Plot the SOC boundary: z = sqrt(x^2 + y^2), 0 <= z <= R
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    r = np.linspace(0.0, R, 65)
    T, Rr = np.meshgrid(theta, r)
    X = Rr * np.cos(T)
    Y = Rr * np.sin(T)
    Z = Rr  # since z = r

    ax.plot_surface(X, Y, Z, color="tab:green", alpha=alpha, rstride=1, cstride=1, linewidth=0)
    ax.text(1.0, 1.0, 1.0, "Second-order cone L^3 (z >= sqrt(x^2+y^2))", color="tab:green")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "figs")
    ensure_output_dir(out_dir)

    # Given vectors in the problem
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 2.0, 1.0])
    diff = tuple((x - y).tolist())  # (-2, 0, 2)

    # Plot for 1(a): R^3_+ and x-y
    ax = setup_3d_ax("Problem 1(a): x - y and R^3_+")
    plot_r3_plus(ax, L=6.0, alpha=0.18)
    draw_vector(ax, diff, color="crimson")
    plt.tight_layout()
    out_path_a = os.path.join(out_dir, "HW12-1.png")
    plt.savefig(out_path_a, dpi=200)
    plt.close(ax.figure)

    # Plot for 1(b): SOC L^3 and x-y
    ax = setup_3d_ax("Problem 1(b): x - y and L^3")
    plot_soc_cone(ax, R=6.0, alpha=0.18)
    draw_vector(ax, diff, color="crimson")
    plt.tight_layout()
    out_path_b = os.path.join(out_dir, "HW12-2.png")
    plt.savefig(out_path_b, dpi=200)
    plt.close(ax.figure)

    print("Saved:", out_path_a)
    print("Saved:", out_path_b)


if __name__ == "__main__":
    main()


