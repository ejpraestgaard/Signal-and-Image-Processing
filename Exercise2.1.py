import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, imshow, subplot, show, title
from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.color import rgb2hsv, hsv2rgb

#1.1: The gamma transformation
def gamma_transform(image, gamma, c = 1):
    image = img_as_float(image)
    return c * image ** gamma
  
#--------------------------------------------------  
  
#1.2: Testing the gamma transform:
I = imread('cameraman.tif', as_gray = True)

I2 = gamma_transform(I, 0.25)
I3 = gamma_transform(I, 0.5)
I4 = gamma_transform(I, 1.5)

#plotting figure
font = {'fontname': 'Times New Roman'}

fig = plt.figure()
fig.suptitle('Figure 1: Various Gamma Corrections', fontsize = 16, **font)

subplot(2,2,1)
imshow(I, cmap = 'gray')
title('Original Image', **font)
axis('off')

subplot(2,2,2)
imshow(I2, cmap = 'gray')
title('Gamma = 0.25', **font)
axis('off')

fig.add_subplot(2,2,3)
imshow(I3, cmap = 'gray')
title('Gamma = 0.5', **font)
axis('off')

fig.add_subplot(2,2,4)
imshow(I4, cmap = 'gray')
title('Gamma = 1.5', **font)
axis('off')

fig.tight_layout()
plt.show()

fig.savefig('Figure1.jpg', dpi=1200)

#1.3: Gamma Correction on RGB 
A = imread('autumn.tif')

#small function, in case we need to reuse it
def rgb_gamma(image, gamma):
    B = np.zeros(A.shape)
    for i in range(3):
         B[:,:,i] = gamma_transform(A[:,:,i], gamma)
    return B

B = rgb_gamma(A, 0.5)

#--------------------------------------------------  

#plotting figure
fig = plt.figure()
fig.suptitle('Figure 2: Gamma Correction for an RGB Image', fontsize = 16, **font)

subplot(211)
imshow(A)
title('Original', **font)
axis('off')

subplot(212)
imshow(B)
title('Gamma Corrected, Gamma = 0.5', **font)
axis('off')

fig.tight_layout()
show()

fig.savefig('Figure2.jpg', dpi=1200)

#--------------------------------------------------  

#1.4: Gamma correction in HSV color space
A = imread('autumn.tif')

A_hsv = rgb2hsv(A)
A_hsv[:,:,2] = gamma_transform(A_hsv[:,:,2], 0.5)
A_rgb = hsv2rgb(A_hsv)

#plotting
fig = plt.figure()
fig.suptitle('Figure 3: Gamma Correction in HSV Colorspace', fontsize = 16, **font)

subplot(211)
imshow(A)
title('Original', **font)
axis('off')

subplot(212)
imshow(A_rgb)
title('Gamma Corrected, with RGB-HSV-RGB Gransform, Gamma = 0.5', **font)
axis('off')

fig.tight_layout()
show()

fig.savefig('Figure3.jpg', dpi=1200)
  
