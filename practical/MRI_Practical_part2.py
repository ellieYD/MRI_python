import matplotlib.pyplot as plt
import numpy
from PIL import Image
import numpy as np
from practical import functions
from scipy import ndimage

def fouriertransform_s1():
    s1 = np.fft.fftshift(np.fft.fft2(functions.s1()))
    plt.figure(1)
    plt.imshow(abs(s1), 'Greys_r')
    plt.title("Fourier transform of s1")
    plt.show()
    return plt.figure(1)

# The result is consistent with the expectation since the image is made out of
# low frequency components, so the result will be close to the origin of the
# k-space

def parta():
    MRIimage = Image.open('D:/Workspace/2021 Year Project/MRI python/resources/head_mri.jpg')
    MRIimage_raw = np.fft.fftshift(np.fft.fft2(MRIimage))
    frequency = np.log(np.abs(MRIimage_raw))
    # the log is used due to the vast majority of the energy is concentrated in
    # the very low spatial frequencies (near the centre of the k-space), without
    # taking the log, there will only be a bright spot in the middle.
    # The bright spot at the origin is the DC offset
    phase = np.angle(MRIimage_raw)

    plt.figure(1)

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
    return plt.figure(1)

def lowpass():
    rect = np.zeros([256, 256])
    rect[112:144, 112:144] = 1
    # low pass rect function
    f1 = Image.open('../resources/head_mri.jpg')
    F1 = np.fft.fftshift(np.fft.fft2(f1))
    F1_lowpass = F1*rect
    f1_lowpass = np.fft.ifft2(np.fft.fftshift(F1_lowpass))

    plt.figure(1)
    plt.gray()
    plt.subplot(141)
    plt.imshow(rect)
    plt.title("Lowpass filter")
    plt.subplot(142)
    plt.imshow(f1)
    plt.title("Original image")
    plt.subplot(143)
    plt.imshow(np.log(np.abs(F1_lowpass)))
    plt.title("Convolution")
    plt.subplot(144)
    plt.imshow(np.abs(f1_lowpass))
    plt.title("Inverse fourier transform")
    plt.show()
    return plt.figure(1)


def highpass():
    rect = np.zeros([256, 256])
    rect[112:144, 112:144] = 1
    rect = 1 - rect
    # high pass rect function
    f1 = Image.open('../resources/head_mri.jpg')
    F1 = np.fft.fftshift(np.fft.fft2(f1))
    F1_highpass = F1*rect
    f1_highpass = np.fft.ifft2(np.fft.fftshift(F1_highpass))

    plt.figure(1)
    plt.gray()
    plt.subplot(141)
    plt.imshow(rect)
    plt.title("Highpass filter")
    plt.subplot(142)
    plt.imshow(f1)
    plt.title("Original image")
    plt.subplot(143)
    plt.imshow(np.log(np.abs(F1_highpass)))
    plt.title("Convolution")
    plt.subplot(144)
    plt.imshow(np.abs(f1_highpass))
    plt.title("Inverse fourier transform")
    plt.show()
    return plt.figure(1)

def halfkspace():
    f1 = Image.open('../resources/head_mri.jpg')
    F1 = np.fft.fftshift(np.fft.fft2(f1))
    F2 = np.log(F1)
    F2[1:256:2, :] = 0
    F1[1:256:2, :] = 0
    # python: start stop step, matlab: start step stop
    F1_half_kspace = F1
    f1_half_kspace = np.fft.ifft2(np.fft.fftshift(F1_half_kspace))
    plt.figure(1)
    plt.gray()
    plt.subplot(131)
    plt.imshow(f1)
    plt.title("Original image")
    plt.subplot(132)
    plt.imshow(np.abs(F1_half_kspace))
    plt.title("Half k space")
    plt.subplot(133)
    plt.imshow(np.abs(f1_half_kspace))
    plt.title("Image domain")
    plt.show()
    return plt.figure(1)
    # This decreases the k space resolution of the image, which means decreased
    # FOV. So the image will be aliased

def shift():
    s4 = functions.s4()
    for i in range (s4[1].size):
        s4[:, i] = np.roll(s4[:, i], -20) #shift in y-axis by 20
    for i in range (s4[1].size):
        s4[i, :] = np.roll(s4[i, :], 40) #shift in x-axis by 40
    s4_fourier = np.fft.fftshift(np.fft.fft2(functions.s4()))
    s4_shifted_fourier = np.fft.fftshift(np.fft.fft2(s4))
    plt.figure(1)
    plt.gray()
    plt.subplot(141)
    plt.imshow(functions.s4())
    plt.title("Original image")
    plt.subplot(142)
    plt.imshow(np.abs(np.log(s4_fourier)))
    plt.title("Fourier transform of the original image")
    plt.subplot(143)
    plt.imshow(s4)
    plt.title("Shifted image")
    plt.subplot(144)
    plt.imshow(np.abs(np.log(s4_shifted_fourier)))
    plt.title("Fourier transform of the shifted image")
    plt.show()
    return plt.figure(1)



def rotation():
    s4_20 = ndimage.rotate(functions.s4(), 20, reshape=False)
    s4_fourier = np.fft.fftshift(np.fft.fft2(functions.s4()))
    s4_20_fourier = np.fft.fftshift(np.fft.fft2(s4_20))
    plt.figure(1)
    plt.gray()
    plt.subplot(141)
    plt.imshow(functions.s4())
    plt.title("Original image")
    plt.subplot(142)
    plt.imshow(np.abs(np.log(s4_fourier)))
    plt.title("Fourier transform of the original image")
    plt.subplot(143)
    plt.imshow(s4_20)
    plt.title("20 degrees rotation")
    plt.subplot(144)
    plt.imshow(np.abs(np.log(s4_20_fourier)))
    plt.title("Fourier transform of the rotated image")
    plt.show()
    return plt.figure(1)

def trim():
    numpy.seterr(divide='ignore')
    s4 = functions.s4()
    for i in range(s4[1].size):
        s4[:, i] = np.roll(s4[:, i], 20)  # shift in y-axis by 20
    for i in range(s4[1].size):
        s4[i, :] = np.roll(s4[i, :], 40)  # shift in x-axis by 40

    s4_fourier = np.fft.fftshift(np.fft.fft2(functions.s4()))
    s4_shifted_fourier = np.fft.fftshift(np.fft.fft2(s4))
    s4_shifted_trimmed_fourier = np.fft.fftshift(np.fft.fft2(s4))
    s4_shifted_trimmed_fourier[:80:1, :] = 0
    s4_shifted_trimmed = np.fft.ifft2(s4_shifted_trimmed_fourier)

    plt.figure(1)
    plt.gray()
    plt.subplot(231)
    plt.imshow(functions.s4())
    plt.title("Original image")

    plt.subplot(234)
    plt.imshow(np.abs(np.log(s4_fourier)))
    plt.title("Fourier transform of the original image")

    plt.subplot(232)
    plt.imshow(s4)
    plt.title("Shifted image")

    plt.subplot(235)
    plt.imshow(np.abs(np.log(s4_shifted_fourier)))
    plt.title("Fourier transform of the shifted image")

    plt.subplot(233)
    plt.imshow(np.abs(s4_shifted_trimmed))
    plt.title("Shifted and trimmed image")

    plt.subplot(236)
    plt.imshow(np.abs(np.log(s4_shifted_trimmed_fourier)))
    plt.title("Fourier transform of the shifted and trimmed")

    plt.show()
    numpy.seterr(divide='warn')
    return plt.figure(1)

if __name__ == "__main__":
    fouriertransform_s1()
    parta()
    lowpass()
    highpass()
    halfkspace()
    shift()
    rotation()
    trim()
