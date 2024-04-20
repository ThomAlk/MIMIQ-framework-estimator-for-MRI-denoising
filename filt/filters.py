from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,denoise_wavelet, estimate_sigma,denoise_nl_means)
import nibabel as nib
import numpy as np

def returnandsave(img,arg):
    
    new_image = nib.Nifti1Image(img, affine=np.eye(4))
    print(new_image)
    nib.save(new_image,arg + ".nii")
    return None

def gaussian(im,path):
    g = estimate_sigma(im, average_sigmas=True)
    img = nd.gaussian_filter(im, sigma=g)
    returnandsave(img,path+'gauss')
    return img    

def bilat(im,path):
    g = estimate_sigma(im, average_sigmas=True)
    img = denoise_bilateral(noisy_img, sigma_spatial=g)
    returnandsave(img,path+'bilat')
    return img

def wavelett(im,path,mode=1):
    if mode==1:
        img = denoise_wavelet(im,method='VisuShrink',mode='soft',rescale_sigma=True)
        returnandsave(img,path+'waveVis')
    else:
        img = denoise_wavelet(im,method='BayesShrink',mode='soft',rescale_sigma=True)
        returnandsave(img,path+'waveBay')
    return img    

def totvar(im,path,weight=1):
    img=denoise_tv_chambolle(noisy_img, weight=1)
    returnandsave(img,path+'totVar')
    return img

def aniso(im,path,n_it=25,kappa=50,gamma=0.02):
    img=anisotropic_diffusion(noisy_img, niter=25, kappa=50, gamma=0.02, option=2) 
    returnandsave(img,path+'aniso')
    return img


def NLM(im,path,size=9,dist=5):
    g = estimate_sigma(im, average_sigmas=True)
    img=denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True,patch_size=g, patch_distance=dist)
    returnandsave(img,path+'NLM')
    return img














