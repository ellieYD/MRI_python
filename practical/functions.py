import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-4, 4, 0.05)
y = np.arange(-4, 4, 0.05)
xx, yy = np.meshgrid(x, y)
def s1():
    s1 = np.exp(-np.pi * (pow(xx, 2) + pow(yy, 2)))
    return s1

def s4():
    condlist = [np.logical_and(np.absolute(xx) <= 0.5, np.absolute(yy) <= 0.5), np.logical_and(np.absolute(xx) > 0.5, np.absolute(yy) > 0.5)]
    choicelist = [1, 0]
    s4 = np.select(condlist, choicelist)
    return s4