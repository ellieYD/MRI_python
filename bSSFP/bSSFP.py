import numpy as np


T1_white = 790
T2_white = 92
T1_CSF = 4*1000
T2_CSF = 2.2*1000

alpha = np.pi/2
rotation_90 = [
    [1,              0,             0],
    [0,  np.cos(alpha), np.sin(alpha)],
    [0, -np.sin(alpha), np.cos(alpha)]
]
rotation_n90 = [
    [1,              0,             0],
    [0,  np.cos(-alpha), np.sin(-alpha)],
    [0, -np.sin(-alpha), np.cos(-alpha)]
]

# two handreds excitations 
excitation = 200
white = np.zeros(3, excitation)
csf = np.zeros(3,excitation)

# initial condition set up










