import numpy as np
from matplotlib import pyplot as plt

x = np.arange(-4, 4, 0.05)
y = np.arange(-4, 4, 0.05)
xx, yy = np.meshgrid(x, y)
def intro():
    s1 = np.exp(-np.pi * (pow(xx, 2) + pow(yy, 2)))
    plt.figure(1)
    plt.contourf(x, y, s1)
    plt.gray()
    plt.title("Initial setup")
    plt.show()

def parta():
    s2 = np.sin(-np.pi * xx) * np.sin(-np.pi * yy) / (pow(np.pi, 2) * xx * yy)
    plt.figure(2)
    plt.contourf(x, y, s2)
    plt.set_cmap('Greys_r')
    plt.title("part a")
    plt.show()

def parte():
    kx_0 = 0.25
    ky_0 = 0.5
    s3 = np.exp(1j * 2 * np.pi * (kx_0 * xx + ky_0 * yy))
    plt.figure(3)
    plt.subplot(121)
    plt.contourf(x, y, s3.real)
    plt.set_cmap('Greys_r')
    plt.title("part e real kx=0.25 ky =0.5")

    plt.subplot(122)
    plt.contourf(x, y, s3.imag)
    plt.set_cmap('Greys_r')
    plt.title("part e imaginary kx=0.25 ky =0.5")
    plt.show()

    kx_1 = 0.25
    ky_1 = 0.75
    s4 = np.exp(1j * 2 * np.pi * (kx_1 * xx + ky_1 * yy))
    plt.figure(3)
    plt.subplot(121)
    plt.contourf(x, y, s4.real)
    plt.set_cmap('Greys_r')
    plt.title("part e real kx=0.25 ky =0.75")

    plt.subplot(122)
    plt.contourf(x, y, s4.imag)
    plt.set_cmap('Greys_r')
    plt.title("part e imaginary kx=0.25 ky =0.75")
    plt.show()

    kx_2 = 0.25
    ky_2 = 1
    s5 = np.exp(1j * 2 * np.pi * (kx_2 * xx + ky_2 * yy))
    plt.figure(4)
    plt.subplot(121)
    plt.contourf(x, y, s5.real)
    plt.set_cmap('Greys_r')
    plt.title("part e real kx=0.25 ky =1")

    plt.subplot(122)
    plt.contourf(x, y, s5.imag)
    plt.set_cmap('Greys_r')
    plt.title("part e imaginary kx=0.25 ky =1")
    plt.show()

    kx_3 = 1
    ky_3 = 1
    s6 = np.exp(1j * 2 * np.pi * (kx_3 * xx + ky_3 * yy))
    plt.figure(5)
    plt.subplot(121)
    plt.contourf(x, y, s6.real)
    plt.set_cmap('Greys_r')
    plt.title("part e real kx=1 ky =1")

    plt.subplot(122)
    plt.contourf(x, y, s6.imag)
    plt.set_cmap('Greys_r')
    plt.title("part e imaginary kx=1 ky =1")
    plt.show()

# kx determines the frequency in x-axis and ky determines the frequency in y-axis in K-space
# kx and ky together determines the angle in the fourier domain.

def rectwin():
    x = np.arange(-4, 4, 0.05)
    y = np.arange(-4, 4, 0.05)
    xx, yy = np.meshgrid(x, y)
    condlist = [np.logical_and(np.absolute(xx) <= 0.5, np.absolute(yy) <= 0.5), np.logical_and(np.absolute(xx) > 0.5, np.absolute(yy) > 0.5)]
    choicelist = [1, 0]
    s4 = np.select(condlist, choicelist)
    plt.figure()
    plt.contourf(x, y, s4)
    plt.set_cmap('Greys_r')
    plt.title("part e rect")
    plt.show()

if __name__ == "__main__":
    intro()
    parta()
    parte()
    rectwin()


