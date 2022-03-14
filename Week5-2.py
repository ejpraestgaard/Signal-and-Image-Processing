#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplot, title, axis, show, imread, imshow
from scipy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift, rfft2
from scipy.signal import gaussian, convolve2d
import time


# In[2]:


def binary_square(x, y, r):
    im = np.zeros([x, y])
    center = x//2 + 1
    im[center - r - 1:center + r, center - r - 1:center + r] = np.full((2*r + 1, 2*r + 1), 255)
    return im


# In[72]:


im = binary_square(15,15,3)


fig0 = plt.figure()
imshow(im,cmap = 'gray')
show()
fig0.savefig('Square.jpg', dpi = 500)

'''
kernel = np.array([[0, 0, 0],[0, 0, 1], [0, 0, 0]])
translation = convolve2d(im, kernel, mode = 'same')
'''


# In[4]:


#imshow(translation, cmap = 'gray')


# In[5]:


def translation(im, num_rows, num_cols, bound = 'wrap'):

    # translates im by num_rows and num_cols by repeated convolution. 
    # By default, we use wrap boundary settings.
    
    up_kernel = np.array([[0, 1, 0],[0, 0, 0], [0, 0, 0]])
    down_kernel = np.array([[0, 0, 0],[0, 0, 0], [0, 1, 0]])

    right_kernel = np.array([[0, 0, 0],[0, 0, 1], [0, 0, 0]])
    left_kernel = np.array([[0, 0, 0],[1, 0, 0], [0, 0, 0]])
    
    for i in range(np.abs(num_rows)):
        if num_rows > 0:
            im = convolve2d(im, up_kernel, mode = 'same',      
                            boundary = '{arg}'.format(arg = bound))
        else:
            im = convolve2d(im, down_kernel, mode = 'same', 
                            boundary = '{arg}'.format(arg = bound))
    for j in range(np.abs(num_cols)):
        if num_cols > 0:
            im = convolve2d(im, right_kernel, mode = 'same', 
                            boundary = '{arg}'.format(arg = bound))
        else:
            im = convolve2d(im, left_kernel, mode = 'same', 
                            boundary = '{arg}'.format(arg = bound))
    return im


# In[6]:


im2 = translation(im, 2, 3, 'wrap')
im3 = translation(im, 10, -2, 'wrap')


# In[7]:


#  Testing the function, with wrap boundary conditions
fig1 = plt.figure()
fig1.set_figheight(10)
fig1.set_figwidth(10)

subplot(131)
imshow(im, cmap = 'gray')
title('White Square')

subplot(132)
imshow(im2, cmap = 'gray')
title('White Square: Up 2, Right 3')

subplot(133)
imshow(im3, cmap = 'gray')
title('White Square: Up 10, Left 2')

show()

fig1.savefig('Figure5.1.jpg', dpi = 500)


# In[8]:


im4 = translation(im, 3, -7, 'symm')
im5 = translation(im, -2, 5, 'symm')


# In[9]:


#  Testing the function, with symmetric boundary conditions
fig2 = plt.figure()
fig2.set_figheight(7)
fig2.set_figwidth(7)

subplot(121)
imshow(im4, cmap = 'gray')
title('White Square: Up 3, Left 7')

subplot(122)
imshow(im5, cmap = 'gray')
title('White Square: Down 2, Right 5')

show()

fig2.savefig('figure5.2.jpg', dpi = 500)


# In[10]:


def fractional_translation(im, x, y, boundary = 'edge'):
    
    # Computes a translation by fractional values, using nearest neigbor interpolation
    
    im = np.pad(im, 3, mode = '{}'.format(boundary))
    output = np.zeros(im.shape)
    
    H = np.array([[1, 0, x], [ 0, 1, y], [0, 0, 1]])
    print('Inverse matrix is \n {}'.format(np.linalg.inv(H)))
    
    for i,j in np.ndenumerate(output[3:-3, 3:-3]):
        inv = np.linalg.inv(H).dot(np.append(i, 1))
        source_pixel = inv[:-1].round().astype(int)
        output[i] = im[source_pixel[0], source_pixel[1]]
        
    return output[3:-3, 3:-3]
            


# In[73]:


#applying the fractional translation
im6 = fractional_translation(im, 0.6, 1.2)

fig3 = plt.figure()
fig3.set_figheight(7)
fig3.set_figwidth(7)

subplot(121)
imshow(im, cmap = 'gray')
title('White Square')

subplot(122)
imshow(im6, cmap = 'gray')
title('White Square: Down 0.6, Right 1.2')

show()

fig3.savefig('Figure5.3.jpg', dpi = 500)


# In[97]:


def fourier_translation(im, x, y):
    
    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(7)
    
    subplot(121)
    imshow(im, cmap = 'gray')
    title('Original')
    
    F_im = fft2(im)
    N, M = im.shape
    
    #subplot(222)
    #imshow(np.log(fftshift(np.abs(F_im)**2)),cmap = 'gray')
    #title('Original Frequency')

    for coord, val in np.ndenumerate(F_im):
        F_im[coord[0], coord[1]] *= np.exp(-1j * 2 * np.pi * (coord[0] * x/N + coord[1] * y/M))
        
    #subplot(223)
    #imshow(np.log(fftshift(np.abs(F_im)**2)),cmap = 'gray')
    #title('Phase-Shifted Frequency')

    inv_F_im = ifft2(F_im)
    
    subplot(122)
    imshow(np.abs(inv_F_im)**2, cmap = 'gray')
    title('Translated Image')
    show()
    
    return fig, np.abs(inv_F_im)**2


# In[98]:


fig4, _ = fourier_translation(im, 3, 3)

fig4.savefig('Figure5.4.jpg')


# In[103]:


path = '/Users/erikpraestgaard/Documents/Signal Processing/Week 1/'

cameraman = imread(path + 'cameraman.tif')

Fcameraman, _ = fourier_translation(cameraman, 100, 100)
Fcameraman.savefig('Translated_Cameraman.jpg', dpi = 500)


# In[107]:


fig5, _ = fourier_translation(im, 1.5, 2.7)
fig5.savefig('Bad_Square.jpg', dpi = 500)

fig6, _ = fourier_translation(cameraman, 100.5, 100.5)
fig6.savefig('Bad_Cameraman.jpg', dpi = 500)


# In[108]:


def fractional_translation(im, x, y, boundary = 'edge'):
    
    # Computes a translation by fractional values, using nearest neigbor interpolation
    
    im = np.pad(im, 3, mode = '{}'.format(boundary))
    output = np.zeros(im.shape)
    
    H = np.array([[1, 0, x], [ 0, 1, y], [0, 0, 1]])
    print('Inverse matrix is \n {}'.format(np.linalg.inv(H)))
    
    for i,j in np.ndenumerate(output[3:-3, 3:-3]):
        inv = np.linalg.inv(H).dot(np.append(i, 1))
        source_pixel = inv[:-1].round().astype(int)
        output[i] = im[source_pixel[0], source_pixel[1]]
        
    return output[3:-3, 3:-3]       


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




