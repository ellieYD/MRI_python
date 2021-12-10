import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
from practical import functions

def fouriertransform_s1():
    s1 =np.fft.fftshift(np.fft.fft2( functions.s1() ))
    plt.figure(1)
    plt.imshow(abs(s1), 'Greys_r')
    plt.title("Fourier transform of s1")
    plt.show()


MRIimage = Image.open('D:/Workspace/2021 Year Project/MRI python/resources/h1.png')
MRIimage_raw = np.fft.fftshift(np.fft.fft2(MRIimage))
frequency = np.log(np.abs(MRIimage_raw))
# the log is used due to the vast majority of the energy is concentrated in
# the very low spatial frequencies (near the centre of the k-space), without
# taking the log, there will only be a bright spot in the middle.
# The bright spot at the origin is the DC offset
phase = np.angle(MRIimage_raw)

plt.figure()

plt.subplot(131)
plt.gray()
plt.title("The original image h1")
plt.imshow(MRIimage)

plt.subplot(132)
plt.imshow(frequency)
plt.title("Its Fourier transform frequency")
plt.gray()
plt.subplot(133)

plt.imshow(phase)
plt.title("Its Fourier transform phase")
plt.gray()
plt.show()

