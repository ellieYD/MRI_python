import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-4, 4, 0.05)
y = np.arange(-4, 4, 0.05)
xx, yy = np.meshgrid(x, y)
def s1():
    s1 = np.exp(-np.pi * (pow(xx, 2) + pow(yy, 2)))
    return s1