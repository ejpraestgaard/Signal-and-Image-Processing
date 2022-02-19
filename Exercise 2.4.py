import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, imshow, subplot, show, title
from scipy.signal import gaussian, convolve2d
from skimage.io import imread, imsave
from skimage.util import img_as_float

font = {'fontname': 'Times New Roman'}

#function for Gaussian noise, using the mean and std grayscale values
def add_gaussian_noise(image):
    image = img_as_float(image)
    noise = np.random.normal(np.mean(image), np.std(image), image.shape)
    return image + noise   

#function for generating 1d Gaussian kernel. f_tau is a 1D Gaussian
def gauss_kernel_1d(sigma, N = None):
    if N == None:
        N = 3 * sigma
    N = int(np.floor(N))
    return gaussian(N, sigma)
  
#function for generating 2d Gaussian kernel.
def gauss_kernel_2d(sigma, N = None):
    if N == None:
        N = 3 * sigma
    N = int(np.floor(N))
    gauss1, gauss2 = gaussian(N, sigma), gaussian(N, sigma)
    return np.outer(gauss1, gauss2)
 
#applied a gaussian to arr
def apply_gaussian(arr, std):
    return np.exp(-(arr**2)/(2 * std**2))

#gets a slice of a matrix arr, with center x,y and with 2*r + 1. If it reaches a boundary, returns a smaller slice
def mat_slice(arr, x, y, radius):
    for i in range(0, radius + 1):
        if x - i >= 0 and y - i >= 0 and x + i < arr.shape[0] and y + i < arr.shape[1]:
            arr_slice = arr[x - i:x + i + 1, y - i:y + i + 1]
    return arr_slice

#First version of the bilateral filter, where the kernels wrap around the boundary
def bilateral_filter(I, sigma, tau, N = None):
    if N == None:
        N = 3 * sigma
    rows, cols = I.shape
    
    delta = lambda x : 0 if x % 2 != 0 else 1
    size = int(np.ceil(N / 2))
    
    filtered_I = np.zeros(I.shape)
    I = np.pad(I, size)

    for y in range(size, cols - size):
        for x in range(size, rows - size):
            
            I_loc = I[x - size:x + size - 1, y - size:y + size - 1]
            photometric_kernel = apply_gaussian(I_loc - I_loc[size, size], tau)
            spatial_kernel = gauss_kernel_2d(sigma, N - delta(N))
            
            omega_ij = np.multiply(spatial_kernel, photometric_kernel)
            prod = np.multiply(omega_ij, I_loc)
            
            filtered_I[x, y] = np.sum(prod) / np.sum(omega_ij)
            
    return filtered_I             
  
#second version, where kernel size is restricted at the boundary
def bilateral_filter2(I, sigma, tau, N = None):
    if N == None:
        N = 3 * sigma
    rows, cols = I.shape
    
    size = int(np.ceil(N / 2))

    filtered_I = np.zeros(I.shape)
    
    for y in range(cols):
        for x in range(rows):
            I_loc = mat_slice(I, x, y, size)
            
            photometric_kernel = apply_gaussian(I_loc - I_loc[I_loc.shape[0] // 2, I_loc.shape[1] // 2], tau)
            spatial_kernel = gauss_kernel_2d(sigma, I_loc.shape[0])
            
            omega_ij = np.multiply(spatial_kernel, photometric_kernel)
            prod = np.multiply(omega_ij, I_loc)
            
            filtered_I[x, y] = np.sum(prod) / np.sum(omega_ij)

    return filtered_I 
  
#loading the image and adding Gaussian noise
I = imread('eight.tif')
I = add_gaussian_noise(I)

#plotting all the kinds of transformations
fig = plt.figure()
fig.suptitle('Bilaterial Filtering for Gaussian Noise', ** font)

subplot(221)
imshow(I, cmap = 'gray')
axis('off')
title('Original with Noise', ** font)

B = bilateral_filter(A, 2, 2)

subplot(222)
imshow(B,cmap = 'gray')
axis('off')
title('Bilateral filter v1, sigma = tau = 2', **font)

C = bilateral_filter2(A, 2, 2)

subplot(223)
imshow(C,cmap = 'gray')
axis('off')
title('Bilateral filter v2, sigma = tau = 2', **font)

D = convolve2d(A, gauss_kernel_2d(2))

subplot(224)
imshow(D,cmap = 'gray')
axis('off')
title('Gauss filter, sigma = 2', **font)
