#!/usr/bin/env python
# coding: utf-8

# In[569]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplot, title, axis, show, imread, imshow
from mpl_toolkits.mplot3d import axes3d
from scipy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift, rfft2
from scipy.signal import gaussian, convolve2d
from scipy.integrate import quad
from skimage.feature import canny, corner_harris, corner_peaks, peak_local_max
from skimage.draw import circle_perimeter
import time
import cv2
import sys
import scipy


# In[712]:


path = '/Users/erikpraestgaard/Documents/Signal Processing/Week 6/'

hand = imread(path + 'hand.tiff')

ims = []
param_vals = [[1, 0, 0], [3, 10, 10], [5, 10, 10], [5, 25, 25]]

ims.append(canny(hand, sigma = 1, low_threshold = 0, high_threshold = 0))
ims.append(canny(hand, sigma = 3, low_threshold = 10, high_threshold = 10))
ims.append(canny(hand, sigma = 5, low_threshold = 10, high_threshold = 10))
ims.append(canny(hand, sigma = 5, low_threshold = 25, high_threshold = 25))



fig0 = plt.figure()
imshow(hand, cmap = 'gray')
axis('off')
show()

fig0.savefig('hand.jpg')


# In[714]:


fig1 = plt.figure()
fig1.set_figwidth(20)

for i in range(len(ims)):
    vals = tuple(param_vals[i])
    subplot(1, len(ims), i + 1)
    imshow(ims[i], cmap = 'gray')
    axis('off')
    title('hand with sigma = %d, low = %d, high = %d' % vals)

show()

fig1.savefig('canny_hand.jpg')


# In[4]:


houses = imread(path + 'modelhouses.png')

fig2 = plt.figure()
imshow(houses, cmap = 'gray')
axis('off')
show()

fig2.savefig('houses.jpg')


# In[183]:


param_vals = [{'method': 'k', 'k': 0.05, 'sigma': 1},
              {'method': 'k', 'k': 0.1, 'sigma': 1},
              {'method': 'k', 'k': 0.05, 'sigma': 10},
              {'method': 'eps', 'eps': 1e-06, 'sigma': 1},
              {'method': 'eps', 'eps': 1e-01, 'sigma': 1}]

fig3 = plt.figure()
fig3.set_figheight(10)
fig3.set_figwidth(20)
fig3.tight_layout()

for i in range(len(param_vals)):
    subplot(2, (len(param_vals) + 1)// 2, i + 1)
    transformed = corner_harris(houses, **param_vals[i])
    imshow(transformed, cmap = 'gray')
    axis('off')
    title('hand with method = %s, method val = %s, sigma = %d' % tuple(param_vals[i].values()))

fig3.subplots_adjust(wspace=0.05)
show()

fig3.savefig('harris_houses.jpg')

fig4 = plt.figure()
fig4.set_figheight(10)
fig4.set_figwidth(20)
fig4.tight_layout()

np.seterr(divide = 'ignore')

for i in range(len(param_vals)):
    subplot(2, (len(param_vals) + 1)// 2, i + 1)
    transformed = corner_harris(houses, **param_vals[i])
    imshow(np.log(transformed), cmap = 'gray')
    axis('off')
    title('hand with method = %s, method val = %s, sigma = %d' % tuple(param_vals[i].values()))

fig4.subplots_adjust(wspace=0.05)
show()

fig4.savefig('harris_log_houses.jpg')


# In[724]:


peaks = corner_peaks(corner_harris(houses, param_vals[0]),min_distance = 5, num_peaks = 500)


# In[725]:


peaks


# In[47]:


houses2 = houses.copy()
houses2 = cv2.cvtColor(houses2, cv2.COLOR_GRAY2RGB)


# In[722]:


fig5 = plt.figure()
imshow(houses2,cmap = 'gray')
plot(peaks[:,0], peaks[:,1], 'r.')
axis('off')
plt.show()

fig5.savefig('Houses_extrema.jpg')


# $$ S = \int_{-\infty}^x \frac{1}{\sqrt{2\pi\sigma^2}} \exp ((-x')^2/2\sigma^2)) dx'$$

# In[360]:


# defining the integrand

s = lambda x_prime : 1/np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-((x_prime ** 2) / (2 * (sigma ** 2))))


# In[375]:


# parameters for varying x

lowbound = -1e2
upbound = 1e2
samples = 100

x = np.linspace(lowbound, upbound, samples)
sample_rate = samples / (upbound - lowbound)


# In[ ]:





# In[ ]:





# In[465]:


# plotting

fig6 = plt.figure()
fig6.set_figwidth(11)
fig6.set_figheight(8)

sigma = 1
S = np.cumsum(s(x)) / sample_rate

subplot(2,3,1)
plt.plot(x,s(x))
title('Graph of the Integrand of S(x,y)')

subplot(2,3,2)
plt.plot(x,S)
title('Graph of S(x), \u03C3 = 1')

I = np.zeros((100, 100))

length, width = I.shape
for i in range(length):
    for j in range(width):
        I[j,i] = S[i]

subplot(2,3,3)
imshow(I)
axis('off')
plt.colorbar()
title('Image S(x,y), \u03C3 = 1')

sigma = 10
S = np.cumsum(s(x)) / sample_rate

subplot(2,3,4)
plt.plot(x,s(x))
title('Graph of the Integrand of S(x,y)')

subplot(2,3,5)
plt.plot(x,S)
title('Graph of S(x), \u03C3 = 10')

I = np.zeros((100, 100))

length, width = I.shape
for i in range(length):
    for j in range(width):
        I[j,i] = S[i]

subplot(2,3,6)
imshow(I)
axis('off')
plt.colorbar()
title('Image S(x,y), \u03C3 = 10')

plt.show()
fig6.savefig('sigma=1.jpg')


# In[ ]:





# In[ ]:





# In[ ]:





# $$ \frac{1}{2\pi\sqrt{\tau^2 + \sigma ^2}}\exp\left(-\frac{x^2}{2(\tau^2 +\sigma ^2)} - \frac{y^2}{2\tau^2}\right), J(0,0,\tau) =\frac{1}{4\pi^2\tau(\tau^2 + \sigma ^2)} $$

# In[495]:


sigma = 1

J00 = lambda tau : 1 / (4 * np.pi ** 2 * tau * (tau ** 2 + sigma ** 2))

tau = np.linspace(1e-2, 10, 100)


# In[496]:


fig7 = plt.figure()

plt.plot(tau, J00(tau))
show()

fig7.savefig('J00.jpg')


# In[428]:


J = lambda x, y, tau : 1 / (2 * np.pi * np.sqrt(tau ** 2 + sigma ** 2))     * np.exp(-(x**2 / (2*(tau ** 2 + sigma ** 2)) + y**2 / (2*(tau ** 2))))


# In[438]:


X, Y = np.mgrid[-10:10:100j, -10:10:100j]
tau = 1


# In[500]:


sigma = 1
tau = [0.1,3,5,10]

Jxy = plt.figure()
Jxy.set_figwidth(11)
Jxy.set_figheight(10)

for i in range(len(tau)):

    func = J(Y, X, tau[i])
    ax = Jxy.add_subplot(2,2,i+1, projection = '3d')
    ax.plot_surface(X,Y, func)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('\u03C4')
    title('Normalized Gradient with \u03C4 = {}'.format(tau[i]))
    
show()

Jxy.savefig('jxy.jpg')


# In[502]:


def gauss_kernel_2d(sigma, N = None):
    if N == None:
        N = 3 * sigma
    N = int(np.floor(N))
    gauss1, gauss2 = gaussian(N, sigma), gaussian(N, sigma)
    return np.outer(gauss1, gauss2)


# In[1169]:


def add_circle(im, center0, center1, r):
    background = np.zeros((2*r+1, 2*r+1))
    rr, cc = circle_perimeter(0,0,r)
    background[rr,cc] = 1
    try:
        im[center0 - r:center0 + r+1, center1 - r:center1 + r+1] += fftshift(background)
    except ValueError:
        pass
    return im


# In[728]:





# In[ ]:





# In[1069]:


hand = imread(path + 'hand.tiff').astype('float') / 255

tau = np.arange(1,10)

peaks = []
vals = []

# finding the peaks

for i in range(len(tau)):
    kernel = gauss_kernel_2d(tau[i], 5)
    xkern, ykern = np.gradient(kernel)

    Jx = convolve2d(hand, xkern, mode = 'same')
    Jy = convolve2d(hand, ykern, mode = 'same')
    
    square_grad = tau[i] * (Jx ** 2 + Jy ** 2)
    tau_peaks = peak_local_max(square_grad, num_peaks = 100)
    peaks.append(tau_peaks)


# In[1112]:


# finding and storing the largest values
idx = np.argpartition(scale_space.ravel(), -200)[-200:]
top_vals = scale_space.ravel()[idx]

# retreiving the radii of the largest values
radii = np.c_[np.unravel_index(idx, scale_space.shape)][:,-1]


# In[1176]:


peaks2 = np.array(peaks)


# In[1158]:


###################################################################################################
### It all goes wrong here. I try to improve my code, but I end up breaking and can't fix it :( ###
### and the undo button on my computer broke somehow, and I am not sure how to fix it :(        ###
###################################################################################################


# In[1179]:


(100 + 2) // 100


# In[1184]:


hand2 = hand.copy()
used = []

for i in peaks2[0:3]:
    for j in i:
        print(j)
    #j = 0
    #radius = (100 + j) // 100
    add_circle(hand2, j[0], j[1], 3 * radius)
    #j += 1
        


# In[1171]:


imshow(hand2)


# In[834]:





# In[824]:


kernel = gauss_kernel_2d(3, 5)
xkern, ykern = np.gradient(kernel)

Jx = convolve2d(hand, xkern, mode = 'same')
Jy = convolve2d(hand, ykern, mode = 'same')
    
square_grad = 3 * (Jx ** 2 + Jy ** 2)
tau_peaks = peak_local_max(square_grad, num_peaks = 50)
vals2 = square_grad[tau_peaks[:,0], tau_peaks[:,1]]


# In[ ]:





# In[ ]:





# In[698]:


hand = imread(path + 'hand.tiff')
floating_hand = hand.astype('float')/255


# In[703]:


imshow(add_circle(floating_hand, 30, 100, 4))


# In[ ]:





# In[ ]:





# In[ ]:




