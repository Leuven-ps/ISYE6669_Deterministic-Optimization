import numpy as np
from scipy.linalg import inv

# Standard form LP:
# min -x1 - 2x2
# s.t. x1 - x2 + x3 = 1
#     -2x1 + x2 + x4 = 1
#      2x1 + x2 + x5 = 5
#      x1, x2, x3, x4, x5 >= 0

# Define matrices
c = np.array([-1, -2, 0, 0, 0])
A = np.array([
    [1, -1, 1, 0, 0],
    [-2, 1, 0, 1, 0],
    [2, 1, 0, 0, 1]
])
b = np.array([1, 1, 5])

print("Original problem:")
print("c =", c)
print("A =")
print(A)
print("b =", b)
print()

# Iteration 1: B = [A3, A4, A5] (columns 2, 3, 4 in 0-based indexing)
print("=== ITERATION 1 ===")
B_indices = [2, 3, 4]  # x3, x4, x5
N_indices = [0, 1]     # x1, x2

B1 = A[:, B_indices]
N1 = A[:, N_indices]
c_B1 = c[B_indices]
c_N1 = c[N_indices]

print("Basis indices:", [f"x{i+1}" for i in B_indices])
print("B =")
print(B1)
print("B^-1 =")
B1_inv = inv(B1)
print(B1_inv)

x_B1 = B1_inv @ b
print("x_B =", x_B1)
print("Basic variables: x3={:.1f}, x4={:.1f}, x5={:.1f}".format(*x_B1))

# Calculate reduced costs
reduced_costs_1 = c_N1 - c_B1.T @ B1_inv @ N1
print("Reduced costs:")
for i, idx in enumerate(N_indices):
    print(f"c̄_{idx+1} = {reduced_costs_1[i]:.3f}")

# Objective value
Z1 = c_B1.T @ x_B1
print(f"Objective value Z = {Z1:.3f}")

# Choose entering variable (most negative, Bland's rule)
entering_idx = 0  # x1 (index 0)
print(f"Entering variable: x{entering_idx+1}")

# Direction vector
A_entering = A[:, entering_idx].reshape(-1, 1)
d_B1 = -B1_inv @ A_entering
print("Direction d_B =", d_B1.flatten())

# Min-ratio test
ratios = []
for i, d_val in enumerate(d_B1.flatten()):
    if d_val < 0:
        ratio = x_B1[i] / (-d_val)
        ratios.append((i, ratio))
        print(f"Ratio for x{B_indices[i]+1}: {x_B1[i]:.1f} / {-d_val:.1f} = {ratio:.3f}")

min_ratio_idx = min(ratios, key=lambda x: x[1])[0]
exiting_var_idx = B_indices[min_ratio_idx]
print(f"Exiting variable: x{exiting_var_idx+1}")
print()

# Iteration 2: B = [A1, A4, A5] (columns 0, 3, 4 in 0-based indexing)
print("=== ITERATION 2 ===")
B_indices = [0, 3, 4]  # x1, x4, x5
N_indices = [1, 2]     # x2, x3

B2 = A[:, B_indices]
N2 = A[:, N_indices]
c_B2 = c[B_indices]
c_N2 = c[N_indices]

print("Basis indices:", [f"x{i+1}" for i in B_indices])
print("B =")
print(B2)
print("B^-1 =")
B2_inv = inv(B2)
print(B2_inv)

x_B2 = B2_inv @ b
print("x_B =", x_B2)
print("Basic variables: x1={:.1f}, x4={:.1f}, x5={:.1f}".format(*x_B2))

# Calculate reduced costs
reduced_costs_2 = c_N2 - c_B2.T @ B2_inv @ N2
print("Reduced costs:")
for i, idx in enumerate(N_indices):
    print(f"c̄_{idx+1} = {reduced_costs_2[i]:.3f}")

# Objective value
Z2 = c_B2.T @ x_B2
print(f"Objective value Z = {Z2:.3f}")

# Choose entering variable
entering_idx = 1  # x2 (index 1)
print(f"Entering variable: x{entering_idx+1}")

# Direction vector
A_entering = A[:, entering_idx].reshape(-1, 1)
d_B2 = -B2_inv @ A_entering
print("Direction d_B =", d_B2.flatten())

# Min-ratio test
ratios = []
for i, d_val in enumerate(d_B2.flatten()):
    if d_val < 0:
        ratio = x_B2[i] / (-d_val)
        ratios.append((i, ratio))
        print(f"Ratio for x{B_indices[i]+1}: {x_B2[i]:.1f} / {-d_val:.1f} = {ratio:.3f}")

min_ratio_idx = min(ratios, key=lambda x: x[1])[0]
exiting_var_idx = B_indices[min_ratio_idx]
print(f"Exiting variable: x{exiting_var_idx+1}")
print()

# Iteration 3: B = [A1, A2, A4] (columns 0, 1, 3 in 0-based indexing)
print("=== ITERATION 3 ===")
B_indices = [0, 1, 3]  # x1, x2, x4
N_indices = [2, 4]     # x3, x5

B3 = A[:, B_indices]
N3 = A[:, N_indices]
c_B3 = c[B_indices]
c_N3 = c[N_indices]

print("Basis indices:", [f"x{i+1}" for i in B_indices])
print("B =")
print(B3)
print("B^-1 =")
B3_inv = inv(B3)
print(B3_inv)

x_B3 = B3_inv @ b
print("x_B =", x_B3)
print("Basic variables: x1={:.1f}, x2={:.1f}, x4={:.1f}".format(*x_B3))

# Calculate reduced costs
reduced_costs_3 = c_N3 - c_B3.T @ B3_inv @ N3
print("Reduced costs:")
for i, idx in enumerate(N_indices):
    print(f"c̄_{idx+1} = {reduced_costs_3[i]:.3f}")

# Objective value
Z3 = c_B3.T @ x_B3
print(f"Objective value Z = {Z3:.3f}")

# Check optimality
if all(r >= 0 for r in reduced_costs_3):
    print("OPTIMAL SOLUTION FOUND!")
    print(f"Optimal solution: x1={x_B3[0]:.1f}, x2={x_B3[1]:.1f}, x3=0, x4={x_B3[2]:.1f}, x5=0")
    print(f"Optimal objective value: {Z3:.1f}")
else:
    print("Solution is not optimal yet.")
    
    # Choose entering variable (x3 has reduced cost -1)
    entering_idx = 2  # x3 (index 2)
    print(f"Entering variable: x{entering_idx+1}")

    # Direction vector
    A_entering = A[:, entering_idx].reshape(-1, 1)
    d_B3 = -B3_inv @ A_entering
    print("Direction d_B =", d_B3.flatten())

    # Min-ratio test
    ratios = []
    for i, d_val in enumerate(d_B3.flatten()):
        if d_val < 0:
            ratio = x_B3[i] / (-d_val)
            ratios.append((i, ratio))
            print(f"Ratio for x{B_indices[i]+1}: {x_B3[i]:.1f} / {-d_val:.1f} = {ratio:.3f}")

    min_ratio_idx = min(ratios, key=lambda x: x[1])[0]
    exiting_var_idx = B_indices[min_ratio_idx]
    print(f"Exiting variable: x{exiting_var_idx+1}")
    print()

    # Iteration 4: B = [A1, A2, A3] (columns 0, 1, 2 in 0-based indexing)
    print("=== ITERATION 4 ===")
    B_indices = [0, 1, 2]  # x1, x2, x3
    N_indices = [3, 4]     # x4, x5

    B4 = A[:, B_indices]
    N4 = A[:, N_indices]
    c_B4 = c[B_indices]
    c_N4 = c[N_indices]

    print("Basis indices:", [f"x{i+1}" for i in B_indices])
    print("B =")
    print(B4)
    print("B^-1 =")
    B4_inv = inv(B4)
    print(B4_inv)

    x_B4 = B4_inv @ b
    print("x_B =", x_B4)
    print("Basic variables: x1={:.1f}, x2={:.1f}, x3={:.1f}".format(*x_B4))

    # Calculate reduced costs
    reduced_costs_4 = c_N4 - c_B4.T @ B4_inv @ N4
    print("Reduced costs:")
    for i, idx in enumerate(N_indices):
        print(f"c̄_{idx+1} = {reduced_costs_4[i]:.3f}")

    # Objective value
    Z4 = c_B4.T @ x_B4
    print(f"Objective value Z = {Z4:.3f}")

    # Check optimality
    if all(r >= 0 for r in reduced_costs_4):
        print("OPTIMAL SOLUTION FOUND!")
        print(f"Optimal solution: x1={x_B4[0]:.1f}, x2={x_B4[1]:.1f}, x3={x_B4[2]:.1f}, x4=0, x5=0")
        print(f"Optimal objective value: {Z4:.1f}")
    else:
        print("Solution is not optimal yet.")