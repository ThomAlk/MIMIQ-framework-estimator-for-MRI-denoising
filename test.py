import filt.filters as flt
import numpy as np
from skimage import (img_as_float,io)
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
import sys
if __name__== '__main__':


    inpath="/home/tpx1/Bureau/algo/test.nii"
    outpath="/home/tpx1/Bureau/algo/"
    img=io.imread(inpath)[0]    
    noisy_img=random_noise(img,mode='Gaussian', rng=None ,clip=True)
    for (im,i) in zip(img,range(len(img))):
        for (col,j) in zip(im,range(len(im))):
            if col>1:
                img[i][j]=1
    t=flt.gaussian(noisy_img,outpath)
    plt.imshow(img,cmap='gray')
    plt.show()
    plt.imshow(t,cmap='gray')
    plt.show()
    plt.imshow(noisy_img,cmap='gray')
    plt.show()
    
    noise_psnr = peak_signal_noise_ratio(img, noisy_img)
    cleaned_psnr = peak_signal_noise_ratio(ref_img, t)


