
import gurobipy as gb

model = gb.read("debug.lp")
model.optimize()

print(model.getVars())
print(model.getConstrs())
print(model.getObjVal())
print(model.getStatus())