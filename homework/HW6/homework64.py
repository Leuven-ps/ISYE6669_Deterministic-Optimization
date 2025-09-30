#!/usr/bin/env python3
"""
HW6 Problem 4: Hill-O-Beans Coffee Company Optimization
Using Gurobi to solve the linear programming problem
"""

import gurobipy as gp
from gurobipy import GRB


def solve_hill_o_beans():
    """Solve the Hill-O-Beans Coffee Company optimization problem."""
    
    # Create a new model
    model = gp.Model("Hill_O_Beans")
    model.setParam('OutputFlag', 1)  # Enable output
    
    # Decision variables
    # X0 = hotel blend, X1 = restaurant blend, X2 = market blend
    X0 = model.addVar(vtype=GRB.CONTINUOUS, name="X0")  # Hotel blend
    X1 = model.addVar(vtype=GRB.CONTINUOUS, name="X1")  # Restaurant blend  
    X2 = model.addVar(vtype=GRB.CONTINUOUS, name="X2")  # Market blend
    
    # Y1 = Abundo, Y2 = Colmado, Y3 = Maximo, Y4 = Saboro
    Y1 = model.addVar(vtype=GRB.CONTINUOUS, name="Y1")  # Abundo
    Y2 = model.addVar(vtype=GRB.CONTINUOUS, name="Y2")  # Colmado
    Y3 = model.addVar(vtype=GRB.CONTINUOUS, name="Y3")  # Maximo
    Y4 = model.addVar(vtype=GRB.CONTINUOUS, name="Y4")  # Saboro
    
    # Set objective function: maximize profit
    # Revenue: 1.25*X0 + 1.50*X1 + 1.40*X2
    # Cost: 0.60*Y1 + 0.80*Y2 + 0.55*Y3 + 0.70*Y4
    model.setObjective(1.25*X0 + 1.50*X1 + 1.40*X2 - 0.60*Y1 - 0.80*Y2 - 0.55*Y3 - 0.70*Y4, GRB.MAXIMIZE)
    
    # Component-blend relationship constraints
    # Y1 = 0.20*X0 + 0.35*X1 + 0.10*X2
    model.addConstr(Y1 == 0.20*X0 + 0.35*X1 + 0.10*X2, "Abundo_usage")
    
    # Y2 = 0.40*X0 + 0.15*X1 + 0.35*X2  
    model.addConstr(Y2 == 0.40*X0 + 0.15*X1 + 0.35*X2, "Colmado_usage")
    
    # Y3 = 0.15*X0 + 0.20*X1 + 0.40*X2
    model.addConstr(Y3 == 0.15*X0 + 0.20*X1 + 0.40*X2, "Maximo_usage")
    
    # Y4 = 0.25*X0 + 0.30*X1 + 0.15*X2
    model.addConstr(Y4 == 0.25*X0 + 0.30*X1 + 0.15*X2, "Saboro_usage")
    
    # Weekly availability constraints
    model.addConstr(Y1 <= 40000, "Abundo_availability")
    model.addConstr(Y2 <= 25000, "Colmado_availability") 
    model.addConstr(Y3 <= 20000, "Maximo_availability")
    model.addConstr(Y4 <= 45000, "Saboro_availability")
    
    # Plant capacity constraint
    model.addConstr(Y1 + Y2 + Y3 + Y4 <= 100000, "Plant_capacity")
    
    # Minimum production requirements
    model.addConstr(X0 >= 10000, "Hotel_minimum")
    model.addConstr(X1 >= 25000, "Restaurant_minimum")
    model.addConstr(X2 >= 30000, "Market_minimum")
    
    # Non-negativity constraints (automatically handled by default bounds)
    
    # Optimize the model
    model.optimize()
    
    # Print results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found!")
        print(f"Optimal objective value: ${model.objVal:,.2f}")
        print("\nOptimal production amounts:")
        print(f"Hotel blend (X0): {X0.x:,.0f} pounds")
        print(f"Restaurant blend (X1): {X1.x:,.0f} pounds") 
        print(f"Market blend (X2): {X2.x:,.0f} pounds")
        print("\nComponent usage:")
        print(f"Abundo (Y1): {Y1.x:,.0f} pounds")
        print(f"Colmado (Y2): {Y2.x:,.0f} pounds")
        print(f"Maximo (Y3): {Y3.x:,.0f} pounds")
        print(f"Saboro (Y4): {Y4.x:,.0f} pounds")
        
        # Calculate total revenue and cost
        total_revenue = 1.25*X0.x + 1.50*X1.x + 1.40*X2.x
        total_cost = 0.60*Y1.x + 0.80*Y2.x + 0.55*Y3.x + 0.70*Y4.x
        print(f"\nTotal revenue: ${total_revenue:,.2f}")
        print(f"Total cost: ${total_cost:,.2f}")
        print(f"Total profit: ${model.objVal:,.2f}")
        
        return model.objVal, X0.x, X1.x, X2.x, Y1.x, Y2.x, Y3.x, Y4.x
    else:
        print("No optimal solution found!")
        print(f"Status: {model.status}")
        return None, None, None, None, None, None, None, None

if __name__ == "__main__":
    solve_hill_o_beans()

