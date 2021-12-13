from PIL import Image
import numpy as np
from numpy import double
import cv2

# RIGHT CLICK on the image NOT LEFT, you can click as many time as you want
# simply close the window for the next instruction.
# The results will be printed out in the terminal
# the window size is of 5 pixels, so be careful when you select the area

poissrnd = np.random.poisson(80,256*256)
poissrnd = np.reshape(poissrnd,(256,256))
right_clicks_signal = []
def mouse_callback(event, x, y, flags, params):
    if event == 2:
        global right_clicks_signal
        right_clicks_signal.append([x, y])

def SNR():
    imgcv = cv2.imread('D:/Workspace/2021 Year Project/MRI python/resources/shepp256.png',0)
    imgcv = cv2.normalize(imgcv.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = Image.open('D:/Workspace/2021 Year Project/MRI python/resources/shepp256.png')
    img = double(img)
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    poissrnd_fourier = np.fft.fftshift(np.fft.fft2(poissrnd))
    img_poiss_fourier = img_fourier+poissrnd_fourier
    img_poiss = np.abs(np.fft.ifft2(img_poiss_fourier))
    scale_width = 640 / imgcv.shape[1]
    scale_height = 480 / imgcv.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(imgcv.shape[1] * scale)
    window_height = int(imgcv.shape[0] * scale)
    cv2.namedWindow('Signal', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Signal', window_width, window_height)
    cv2.setMouseCallback('Signal', mouse_callback)
    cv2.imshow('Signal', imgcv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    window = 5
    signal = np.empty(shape=(0, 0))
    for xy in right_clicks_signal:
        [y, x] = xy
        signal_piece = img_poiss[x:x+window, y:y+window]
        signal = np.append(np.reshape(signal_piece,(1, signal_piece.size)),signal)
    right_clicks_signal.clear()
    signal_mean = np.mean(signal)

    cv2.namedWindow('Noise', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Noise', window_width, window_height)
    cv2.setMouseCallback('Noise', mouse_callback)
    cv2.imshow('Noise', imgcv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    noise = np.empty(shape=(0, 0))
    for xy in right_clicks_signal:
        [y, x] = xy
        noise_piece = img_poiss[x:x + window, y:y + window]
        noise = np.append(np.reshape(noise_piece, (1, noise_piece.size)), noise)
    backgourd_std = np.std(noise)
    right_clicks_signal.clear()
    SNR_tissue = signal_mean / backgourd_std
    print("SNR of tissue: ",SNR_tissue)




def CNR():
    imgcv = cv2.imread('D:/Workspace/2021 Year Project/MRI python/resources/shepp256.png',0)
    imgcv = cv2.normalize(imgcv.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = Image.open('D:/Workspace/2021 Year Project/MRI python/resources/shepp256.png')
    img = double(img)
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    poissrnd_fourier = np.fft.fftshift(np.fft.fft2(poissrnd))
    img_poiss_fourier = img_fourier+poissrnd_fourier
    img_poiss = np.abs(np.fft.ifft2(img_poiss_fourier))
    scale_width = 640 / imgcv.shape[1]
    scale_height = 480 / imgcv.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(imgcv.shape[1] * scale)
    window_height = int(imgcv.shape[0] * scale)
    cv2.namedWindow('tissue1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tissue1', window_width, window_height)
    cv2.setMouseCallback('tissue1', mouse_callback)
    cv2.imshow('tissue1', imgcv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    window = 5
    tissue1 = np.empty(shape=(0, 0))
    for xy in right_clicks_signal:
        [y, x] = xy
        tissue1_piece = img_poiss[x:x+window, y:y+window]
        tissue1 = np.append(np.reshape(tissue1_piece,(1, tissue1_piece.size)),tissue1)
    right_clicks_signal.clear()
    tissue1_mean = np.mean(tissue1)

    cv2.namedWindow('tissue2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('tissue2', window_width, window_height)
    cv2.setMouseCallback('tissue2', mouse_callback)
    cv2.imshow('tissue2', imgcv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    window = 5
    tissue2 = np.empty(shape=(0, 0))
    for xy in right_clicks_signal:
        [y, x] = xy
        tissue2_piece = img_poiss[x:x+window, y:y+window]
        tissue2 = np.append(np.reshape(tissue2_piece,(1, tissue2_piece.size)),tissue2)
    right_clicks_signal.clear()
    tissue2_mean = np.mean(tissue2)

    cv2.namedWindow('Noise', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Noise', window_width, window_height)
    cv2.setMouseCallback('Noise', mouse_callback)
    cv2.imshow('Noise', imgcv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    noise = np.empty(shape=(0, 0))
    for xy in right_clicks_signal:
        [y, x] = xy
        noise_piece = img_poiss[x:x + window, y:y + window]
        noise = np.append(np.reshape(noise_piece, (1, noise_piece.size)), noise)
    backgourd_std = np.std(noise)
    right_clicks_signal.clear()
    CNR_tissue = np.abs((tissue1_mean-tissue2_mean) / backgourd_std)
    print("CNR: ",CNR_tissue)

if __name__ == "__main__":
    SNR()
    SNR()
    CNR()