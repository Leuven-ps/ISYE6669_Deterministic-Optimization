\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xe10abdb866b8611b
Minimize
  10 p[1] + 5 p[3] + 8 p[5]
Subject To
 ref_bus: theta[1] = 0
 p_1_min: p[1] >= 20
 p_1_max: p[1] <= 270
 p_1_balance: - x[1,2] + x[6,1] + p[1] = 0
 p_3_min: p[3] >= 20
 p_3_max: p[3] <= 250
 p_3_balance: x[2,3] - x[3,4] + p[3] = 0
 p_5_min: p[5] >= 10
 p_5_max: p[5] <= 300
 p_5_balance: x[4,5] - x[5,6] + p[5] = 0
 d_2_balance: x[1,2] - x[2,3] = 120
 d_4_balance: x[3,4] - x[4,5] = 95
 d_6_balance: x[5,6] - x[6,1] = 105
 f_(1,_2)_min: x[1,2] >= -100
 f_(1,_2)_max: x[1,2] <= 100
 theta_(1,_2)_nodal: x[1,2] - 11.6 theta[1] + 11.6 theta[2] = 0
 f_(2,_3)_min: x[2,3] >= -120
 f_(2,_3)_max: x[2,3] <= 120
 theta_(2,_3)_nodal: x[2,3] - 5.9 theta[2] + 5.9 theta[3] = 0
 f_(3,_4)_min: x[3,4] >= -50
 f_(3,_4)_max: x[3,4] <= 50
 theta_(3,_4)_nodal: x[3,4] - 13.7 theta[3] + 13.7 theta[4] = 0
 f_(4,_5)_min: x[4,5] >= -90
 f_(4,_5)_max: x[4,5] <= 90
 theta_(4,_5)_nodal: x[4,5] - 9.8 theta[4] + 9.8 theta[5] = 0
 f_(5,_6)_min: x[5,6] >= -60
 f_(5,_6)_max: x[5,6] <= 60
 theta_(5,_6)_nodal: x[5,6] - 5.6 theta[5] + 5.6 theta[6] = 0
 f_(6,_1)_min: x[6,1] >= -50
 f_(6,_1)_max: x[6,1] <= 50
 theta_(6,_1)_nodal: x[6,1] + 10.5 theta[1] - 10.5 theta[6] = 0
Bounds
 x[1,2] free
 x[2,3] free
 x[3,4] free
 x[4,5] free
 x[5,6] free
 x[6,1] free
 theta[1] free
 theta[2] free
 theta[3] free
 theta[4] free
 theta[5] free
 theta[6] free
End
