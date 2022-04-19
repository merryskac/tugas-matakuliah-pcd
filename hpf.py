import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Read images
img = cv2.imread('spbob1.jpg', 0)

#  Fourier transform
fimg = np.fft.fft2(img)
fshift = np.fft.fftshift(fimg)

#  Set the high pass filter
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

#  Inverse Fourier transform
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

#  Display original image and high pass filter processing image
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('gray Image')
plt.axis('off')
plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title('Result Image')
plt.axis('off')
plt.show()