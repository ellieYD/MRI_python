import math
from cgitb import grey

import np as np
import pydicom as pydicom
import matplotlib.pyplot as plt

# Create a discrete time signal x[n]

sampling_frequency = 100
discrete_signal_size = 1000
discrete_time_axis = np.arange(0, discrete_signal_size, 1/sampling_frequency)

discrete_signal = np.sinc(discrete_time_axis)

plt.subplot(1, 2, 1)
plt.plot(discrete_time_axis, discrete_signal)



# discrete time fourier transform x[k] = \sum x[n] exp(-1j 2 pi n k /N)

sample_size = 100
discrete_fourier_transform = (np.zeros(sample_size, dtype = complex))


for k in range(sample_size):
    for n in range(sample_size):
        discrete_fourier_transform[k] = discrete_fourier_transform[k] + discrete_signal[n]*np.exp(-1j*2*np.pi*n*k/sample_size)

plt.subplot(1, 2, 2)
DFT = np.abs(np.real(discrete_fourier_transform))
discrete_fourier_transform_axis = np.arange(-sample_size/2, sample_size/2, 1)
DFT_shifted = discrete_fourier_transform

mid = int(sample_size/2)
if mid%2 == 0:
    for i in range(mid):
        temp = DFT_shifted[i]
        DFT_shifted[i] = DFT_shifted[mid+i]
        DFT_shifted[mid+i] = temp

else:
    for i in range(mid):
        temp = DFT_shifted[i]
        DFT_shifted[i] = DFT_shifted[mid+i]
        DFT_shifted[mid+i+1] = temp


plt.plot(discrete_fourier_transform_axis, DFT_shifted)
width = 10
sinc_time = np.arange(-width/2, width/2, 0.001)
sinc = np.sinc(sinc_time)


plt.figure()
plt.subplot(1, 2, 1)
reconstruct = np.convolve(DFT_shifted,sinc)
plt.plot(reconstruct)

plt.subplot(1, 2, 2)
signal = np.fft.ifft(abs(reconstruct))
plt.plot(signal)
plt.show()



