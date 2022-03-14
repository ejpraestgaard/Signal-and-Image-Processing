def fourier_translation(im, x, y):
    
    '''
    imshow(im, cmap = 'gray')
    show()
    
    F_im = fft2(im)
    N = im.shape[0]//2
    
    imshow(fftshift(np.abs(F_im)**2),cmap = 'gray')
    show()
    '''
    #method 1
    
    side = int(np.sqrt(im.size))
    mask = np.zeros((side//2, side//2))

    for u in range(side//2):
        for v in range(side//2):
            #print(np.exp(-1j * 2 * np.pi * (u * x + v * y)/N))
            mask[u,v] = np.exp(-1j * 2 * np.pi * (u * x + v * y)/ N)
            
    F_im = np.matmul(F_im, pad_mask)        
       
    #method 2
    #for i,j in np.ndenumerate(F_im):
    #    F_im[i[0], i[1]] *= np.exp(-1j * 2 * np.pi * (i[0] * x + i[1] * y)/N)

    #imshow(fftshift(np.abs(F_im)**2),cmap = 'gray')
    #show()
    
    inv_F_im = ifft2(F_im)
    
    #imshow(fftshift(np.abs(inv_F_im)**2), cmap = 'gray')
    #show()

    return np.abs(inv_F_im)**2
