import cv2
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from numpy import double

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

def mouse_callback(event, x, y, flags, params):
    if event == 2:
        global right_clicks_signal
        right_clicks_signal.append([x, y])




