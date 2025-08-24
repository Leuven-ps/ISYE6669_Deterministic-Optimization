"""
Portfolio optimization with CVXPY
See examples at http://cvxpy.org
Author: Shabbir Ahmed
"""

import pandas as pd
import numpy as np
import cvxpy as cp

mp = pd.read_csv("monthly_prices_HW1.csv",index_col=0) #update to match your path

mr = pd.DataFrame()

for s in mp.columns:
    date = mp.index[0]
    pr0 = mp[s][date] 
    for t in range(1,len(mp.index)):
        date = mp.index[t]
        pr1 = mp[s][date]
        ret = (pr1-pr0)/pr0
        mr.at[date,s]=ret
        pr0 = pr1
        
symbols = mr.columns
return_data = mr.values.T
r = np.mean(return_data, axis=1).astype(float)

C = np.cov(return_data).astype(float)
C = 0.5*(C + C.T)
print(C)

# output
print("----------------------")
for j in range(len(symbols)):
    print('%s: Exp ret = %f, Risk = %f' %(symbols[j],r[j], C[j,j]**0.5))
   

###### optimization model ######
n = len(symbols)
x = cp.Variable(n)
req_return = 0.02
ret = r.T@x
risk = cp.quad_form(x, C)
objective = cp.Minimize(risk)
constraints = [cp.sum(x) == 1, ret >= req_return, x >= 0]
problem = cp.Problem(objective, constraints)

# print output
try:
    #problem.solve(solver = 'CVXOPT')
    problem.solve(solver=cp.SCS)
    print("----------------------")
    print("Optimal portfolio")
    print("----------------------")
    for s in range(len(symbols)):
        print('x[%s] = %f'%(symbols[s],x.value[s]))
    print("----------------------")
    print('Exp ret = %f' %(ret.value))
    print('risk    = %f' %((risk.value)**0.5))
    print("----------------------")
except:
    print('Error')