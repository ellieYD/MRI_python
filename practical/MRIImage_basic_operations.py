import math
from cgitb import grey

import np as np
import pydicom as pydicom
import matplotlib.pyplot as plt

import matplotlib.image as img
# read in the file
# put any mri data in dicom
MRIimage_raw = pydicom.read_file('D:/Workspace/2021 Year Project/MRI python/resources/MR000001')


#read the dicom file
MRIimage = MRIimage_raw.pixel_array

# Two dimensional FFT
fft = np.fft.fft2(MRIimage)

#shift the zero frequency to the center
centerfft = np.fft.fftshift(fft)

frequency = np.log(np.abs(centerfft))
phase = np.angle(centerfft)


real = np.exp(frequency)*np.cos(phase);
imag = np.exp(frequency)*np.sin(phase);

#size of the matrix
size = (int(math.sqrt(real.size)))

matrix = np.zeros([size, size], dtype=complex)
matrix.real = np.array(real)
matrix.imag = np.array(imag)

fshift = np.fft.ifftshift(matrix)
MRI = np.fft.ifft2(fshift)

#convert into grey scale
plt.subplot(141), plt.imshow(MRIimage, 'Greys_r'), plt.title("Original MRI Image")
plt.subplot(142), plt.imshow(frequency, 'Greys_r'), plt.title("Frequency")
plt.subplot(143), plt.imshow(phase, 'Greys_r'), plt.title("Phase")
plt.subplot(144), plt.imshow(np.abs(MRI), 'Greys_r'), plt.title("Recovered")
plt.show()




