#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplot, title, axis, show, imread, imshow
import skimage
from skimage.feature import canny, peak_local_max
import skimage.transform
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from scipy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift, rfft2
from skimage.draw import circle_perimeter
import time
import cv2
from cv2 import circle
import sys
import scipy
import math

from scipy.signal import gaussian, convolve, convolve2d
from scipy.ndimage import gaussian_filter


# In[3]:


def binary_square(x, y, r):
    im = np.zeros([x, y])
    center = x//2 + 1
    im[center - r - 1:center + r, center - r - 1:center + r] = np.full((2*r + 1, 2*r + 1), 255)
    return im


# In[238]:


im = binary_square(100,100,3)


fig0 = plt.figure()
imshow(im,cmap = 'gray')
show()
#fig0.savefig('Square.jpg', dpi = 500)

'''
kernel = np.array([[0, 0, 0],[0, 0, 1], [0, 0, 0]])
translation = convolve2d(im, kernel, mode = 'same')
'''


# In[ ]:





# In[260]:


def nearestNeighborTranslation(im, t1, t2, c1, c2, s, theta):
    
    H = np.array([[s * np.cos(theta), -s * np.sin(theta), -(c1 * s * np.cos(theta) - c2 * s * np.sin(theta)) + t1 + c1], 
                  [s * np.sin(theta), s * np.cos(theta), -(c1 * s * np.sin(theta) + c2 * s * np.cos(theta)) + t1 + c1], 
                  [0, 0, 1]])

    return skimage.transform.warp(im, np.linalg.inv(H), order = 0, mode = 'wrap')


# In[262]:


transformed = nearestNeighborTranslation(im, t1 = 15.7,  t2 = 10.4, 
                                         c1 = im.shape[0] // 2, c2 = im.shape[1] // 2, 
                                         s = 2, theta = np.pi / 10)
fig = plt.figure()
plt.subplot(1,2,1)
imshow(im, cmap = 'gray', origin = 'lower')
axis('off')
title('Original Square')
plt.subplot(1,2,2)
imshow(transformed, cmap = 'gray', origin = 'lower')
axis('off')
title('Transformed')

show()
fig.savefig('Squares.png')


# In[223]:


def GaussKernel(sigma, N = None):
    if N == None:
        N = 5 * sigma
    N = int(np.floor(N))
    gauss1, gauss2 = gaussian(N, sigma), gaussian(N, sigma)
    return np.outer(gauss1, gauss2)


# In[ ]:





# In[790]:


path = '/Users/erikpraestgaard/Documents/Signal Processing/Week 8/'
im = imread(path + 'modelhouses.png').astype('float32')


# In[791]:


# testing

points = peak_local_max(R, num_peaks = 350, min_distance = 5)
im2 = im.copy()

harris = skimage.feature.corner_harris(im2)

imshow(harris,cmap = 'gray')
plt.colorbar()

harris


# In[792]:


find_spatial_max(scaleSpace)


# In[793]:


def find_spatial_max(L):
    for l in range(0, L.shape[0]):
        p_max=peak_local_max(L[0,:,:], min_distance = 5)
        p_max = np.hstack([l*np.ones((p_max.shape[0],1), dtype=int), p_max] )
        if l==0:
            p_maxi = p_max
        else: 
            p_maxi = np.vstack([p_maxi, p_max])
    return p_maxi


# In[794]:


def find_scale_max(L, p_maxi):
    keep = np.zeros((p_maxi.shape[0],), dtype=bool)
    for i in range(1,p_maxi.shape[0]):
        p = p_maxi[i,:]
        #print(p)
        Lcenter = L[p[0], p[1], p[2]]
        if p[2] > 0:
            Lup = L[p[0], p[1], p[2]-1]
        else:
            Lup = -np.inf
            
        if p[2] < L.shape[2]-1:
            Ldown = L[p[0], p[1], p[2]+1]
        else:
            Ldown = -np.inf
        
        if Lcenter > Lup and Lcenter > Ldown: 
            keep[i] = True
    
    return p_maxi[keep,:]
    
    
def find_scalespace_max(L):

    p_maxi = find_spatial_max(L)
    p_maxi = find_scale_max(L, p_maxi)
        
    return p_maxi
    
def keep_strongest_points(L, points, num_p):
    
    sorted_idx = L[points[:,0], points[:,1], points[:,2]].argsort()
    if sorted_idx.shape[0] >= num_p:
        strongest_idx = sorted_idx[-(num_p+1):-1]
    else:
        strongest_idx = sorted_idx
        
    return points[np.flip(strongest_idx)]

def detect_scalespace_edges(L, num_p = 20):
    
    p_maxi = find_scalespace_max(L)
    p_maxi = keep_strongest_points(L, p_maxi, num_p)
    
    return p_maxi


# In[807]:


# generating state space

scale_levels = 30
vals = np.logspace(0, 5, scale_levels , base=2)

scaleSpace = []
alpha = 0.001
k = 3

for sigma in vals:
       
    Ix = gaussian_filter(im, sigma, order = (0,1))
    Iy = gaussian_filter(im, sigma, order = (1,0))

    Ixx = gaussian_filter(np.multiply(Ix,Ix), k * sigma)
    Ixy = gaussian_filter(np.multiply(Ix,Iy), k * sigma)
    Iyy = gaussian_filter(np.multiply(Iy,Iy), k * sigma)
    
    #Ixx = gaussian_filter(gaussian_filter(im, sigma, (0,2)), k * sigma)
    #Iyy = gaussian_filter(gaussian_filter(im, sigma, (2,0)), k * sigma)

    tr = Ixx + Iyy
    det = np.multiply(Ixx, Iyy) - np.multiply(Ixy, Ixy)
    

    R = sigma ** 4 * (det - alpha * (tr ** 2))
    scaleSpace.append(R)
    
scaleSpace = np.asarray(scaleSpace)


# In[808]:


scaleSpace.shape


# In[809]:


corners = detect_scalespace_edges(scaleSpace, 350)


# In[810]:


len(corners)


# In[811]:


im2 = im.copy().astype('float32')
im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB)


# In[812]:


for i in corners:
    print((i[1], i[2]))
    circle(im2, (i[1], i[2]), radius = i[0] + 1, color = (1, 0, 0), thickness = 1)
    
imshow(im2)
axis('off')
plt.savefig('housecorners2.jpg')


# In[779]:


coins = imread(path + 'coins.png').astype('float32')
euros = imread(path + 'overlapping_euros1.png').astype('float32')


# In[629]:


# Compute Image histogram. From HW 2

def bar_plot(xs, ys, title=None, filepath=None):
    fig, ax = plt.subplots()
    ax.bar(xs, ys, color='#47476b', width=1.0)
    ax.set_facecolor("#f0f0f5")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    if title:
        ax.set_title(title)
    if filepath:
        fig.savefig(filepath, bbox_inches='tight')
    #plt.show()
    return fig

def compute_cumu_hist(hist):
    return np.cumsum(hist)/np.sum(hist)


# In[654]:


# plot histograms

fig, ax = plt.subplots(1,2)

histCoins, _ = np.histogram(coins, bins=255)
ax[0] = bar_plot(np.arange(0,255), histCoins, title = 'Coins Histogram')
histEuros, _ = np.histogram(euros, bins=255)
ax[1] = bar_plot(np.arange(0,255), histEuros, title = 'Euros Histogram')
show()
fig.savefig('hist.jpg')


# In[655]:


from scipy.signal import find_peaks


# In[ ]:





# In[787]:


def findThreshold(im):
    
    # Option to smooth the images, hence smooth the histograms
    #im = gaussian_filter(im, sigma = 0.25)
    
    # make histogram
    hist, _ = np.histogram(im, bins = 255)
    
    # finds the two hightest peaks of the image histogram
    
    peaks, heights = find_peaks(hist, distance = 50, height = 200)
    idx = np.argpartition(heights['peak_heights'], -2)[-2:]
    
    # finds the threshold by finding the threshold index in the CDF, then finding that 
    # CDF value in the sorted original image
    
    thresholdIdx = int(np.mean(peaks[idx]))
    thresholdIdxCDF = np.cumsum(hist)[thresholdIdx]
    threshold = np.sort(im.ravel())[thresholdIdxCDF]
    
    print('Threshold Value is', threshold)
    return threshold


# In[776]:


def segmentIm(im, threshold):
    segmented = im.copy()
    for x in range(segmented.shape[0]):
        for y in range(segmented.shape[1]):
            if segmented[x,y] > threshold:
                segmented[x,y] = 1
            else:
                segmented[x,y] = 0
                
    return segmented


# In[788]:


pics = plt.figure(figsize = (10, 5))

subplot(1,2,1)
imshow(euros, cmap = 'gray')
axis('off')
title('Euros')

subplot(1,2,2)
imshow(coins, cmap = 'gray')
axis('off')
title('Coins')

show()

pics.savefig('CoinsAndEuros.jpg')


# In[789]:


seg = plt.figure(figsize = (10, 5))

subplot(1,2,1)
imshow(segmentIm(euros, findThreshold(euros)), cmap = 'gray')
axis('off')
title('Segmented Euros')

subplot(1,2,2)
imshow(segmentIm(coins, findThreshold(coins)), cmap = 'gray')
axis('off')
title('Segmented Coins')

show()
seg.savefig('segmented.jpg')


# In[999]:


#def LSIDegredation(im, h, n):
#    return convolve2d(im, h, mode = 'same') + n

def LSIDegredation(im, h, n):
    F = fft2(im)
    H = fft2(h)
    N = fft2(n)
    G = F*H + N
    return ifftshift(np.abs(ifft2(G) ** 2))


# In[1079]:


path2 = '/Users/erikpraestgaard/Documents/Signal Processing/Week 1/'
trui = imread(path2 + 'trui.png').astype('float32')

h1 = GaussKernel(sigma = 3, N = trui.shape[0])
n1 = np.random.normal(loc = 0, scale = 5, size = (trui.shape))

h2 = GaussKernel(sigma = 1, N = trui.shape[0])
n2 = np.random.gamma(shape = 1, scale = 0.25, size = (trui.shape))

n3 = np.random.rayleigh(scale = 1, size = (trui.shape))


# In[1081]:


LSI = plt.figure(figsize = (15, 5))
subplot(1,3,1)
imshow(LSIDegredation(trui, h1, n1), cmap = 'gray')
axis('off')
title('G(3) Filter and White Noise, std = 5')

subplot(1,3,2)
imshow(LSIDegredation(trui, h2, n2), cmap = 'gray')
axis('off')
title('G(1) Filter and Gamma Noise, std = 0.25')

subplot(1,3,3)
imshow(LSIDegredation(trui, h2, n3), cmap = 'gray')
axis('off')
title('G(1) Filter with Rayleigh Noise, std = 1')

LSI.savefig('LSI.jpg')


# In[1082]:


def DIF(im, psf, epsilon = 0.01):
    # computes the Fourier transform. 
    # We need to add epsilon to account for 0 values in the PSF
    F = fft2(im)
    H = fft2(psf) + epsilon
    return fftshift(np.abs(ifft2(np.divide(F,H)) ** 2))


# In[ ]:





# In[1084]:


noisy1 = LSIDegredation(trui, h1, n1)
noisy2 = LSIDegredation(trui, h2, n3)

h3 = GaussKernel(sigma = 1, N = trui.shape[0])
n4 = np.zeros(trui.shape)

noisy3 = LSIDegredation(trui, h3, n4)

DIFs = plt.figure(figsize = (15, 5))

ims = [DIF(noisy1, h1), DIF(noisy2, h2), DIF(noisy3, h3)]

subplot(1,3,1)
imshow(ims[0], cmap = 'gray')
axis('off')
title('DIF with g(1) and White Noise, std = 5')


subplot(1,3,2)
imshow(ims[1], cmap = 'gray')
axis('off')
title('DIF with g(3) and Gamma Noise, std = 0.25')

subplot(1,3,3)
imshow(ims[2], cmap = 'gray')
axis('off')
title('DIF with g(1) and No Noise')

DIFs.savefig('DIFS.jpg')


# In[ ]:





# In[1085]:


def Weiner(im, psf, k, epsilon = 0.01):
    F = fft2(im)
    H = fft2(psf) + epsilon
    Weiner = 1 / H * ((H * np.conj(H)) / (H * np.conj(H) + k))
    recovered = ifft2(np.multiply(F, Weiner))
    return ifftshift(np.abs(recovered) ** 2)


# In[1116]:


fig = plt.figure(figsize = (20, 20))

subplot(3,3,1)
imshow(Weiner(noisy1, h1, 10), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 5, k = 10')

subplot(3,3,2)
imshow(Weiner(noisy1, h1, 100), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 5, k = 100')

subplot(3,3,3)
imshow(Weiner(noisy1, h1, 1000), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 5, k = 1000')

subplot(3,3,4)
imshow(Weiner(noisy2, h2, 10), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) Guassian Noise with std 0.25 k = 10')

subplot(3,3,5)
imshow(Weiner(noisy2, h2, 100), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 0.25, k = 100')

subplot(3,3,6)
imshow(Weiner(noisy2, h2, 1000), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 0.25, k = 1000')

subplot(3,3,7)
imshow(Weiner(noisy3, h3, 10), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) and no Noise,  k = 10')

subplot(3,3,8)
imshow(Weiner(noisy3, h3, 100), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) and no Noise, k = 100')

subplot(3,3,9)
imshow(Weiner(noisy3, h3, 1000), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) and no Noise, k = 1000')

fig.savefig('Weiner1.jpg')


# In[1114]:


fig = plt.figure(figsize = (20, 5))

subplot(1,3,1)
imshow(Weiner(noisy2, h2, 10), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) Guassian Noise with std 0.25 k = 10')

subplot(1,3,2)
imshow(Weiner(noisy2, h2, 100), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 0.25, k = 100')

subplot(1,3,3)
imshow(Weiner(noisy2, h2, 1000), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) White Noise with std 0.25, k = 1000')

fig.savefig('Weiner2.jpg')


# In[1113]:


fig = plt.figure(figsize = (20, 5))

subplot(1,3,1)
imshow(Weiner(noisy3, h3, 10), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) and no Noise,  k = 10')

subplot(1,3,2)
imshow(Weiner(noisy3, h3, 100), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) and no Noise, k = 100')

subplot(1,3,3)
imshow(Weiner(noisy3, h3, 1000), cmap = 'gray')
axis('off')
title('Weiner Inverse with g(1) and no Noise, k = 1000')

fig.savefig('Weiner3.jpg')


# In[ ]:




