from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,denoise_wavelet, estimate_sigma,denoise_nl_means)
import nibabel as nib
import numpy as np
import warnings
from medpy.filter.smoothing import anisotropic_diffusion
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from MIMIQ.parameters import Parameters
import sys


class Filters:
    def __init__(self):
        self.parameters = Parameters()
        self.filters = [self.gaussian, self.bilat, self.wavelett_visu, self.totvar, self.aniso, self.nlm, self.wavelett_bayes]
        self.update_parameters()

    def update_parameters(self):
        self.sigma = self.parameters.gaussian_sigma
        self.w = self.parameters.totvar_weight
        self.n_it = self.parameters.aniso_iter
        self.kappa = self.parameters.aniso_kappa
        self.gamma = self.parameters.aniso_gamma
        self.size = self.parameters.NLM_size
        self.dist = self.parameters.NLM_dist

    def gaussian_imgs(self, img, sigmas):
        def gaussian(img, sig):
            return nd.gaussian_filter(img, sigma=sig)
        params = []
        img_params = []
        for s in sigmas:
            imtemp = gaussian(img, sig=s)
            e = peak_signal_noise_ratio(img, imtemp)
            img_params.append(imtemp)
            params.append({'sigma': s, 'psnr': e})
        return img_params, params

    def totvar_imgs(self, img, weights):
        def totvar(img, w):
            return denoise_tv_chambolle(img, weight=w)
        params = []
        img_params = []
        for w in weights:
            imtemp = totvar(img, w)
            e = peak_signal_noise_ratio(img, imtemp)
            img_params.append(imtemp)
            params.append({'weight': w, 'psnr': e})
        return img_params, params

    def aniso_imgs(self, img, n_iters, kappas, gammas):
        def aniso(img, n_it, kappa, gamma):
            return anisotropic_diffusion(img, niter=n_it, kappa=kappa, gamma=gamma, option=2)
        params = []
        img_params = []
        for k in kappas:
            for g in gammas:
                imtemp = aniso(img, n_it=n_iters, kappa=k, gamma=g)
                e = peak_signal_noise_ratio(img, imtemp)
                img_params.append(imtemp)
                params.append({'kappa': k, 'gamma': g,'iter': n_iters,'psnr': e})
        return img_params, params

    def nlm_imgs(self, img, sizes, dists):
        def nlm(img, size, dist):
            g = estimate_sigma(img, average_sigmas=True)
            return denoise_nl_means(img, h=1.15 * g, fast_mode=True, patch_size=size, patch_distance=dist)
        params = []
        img_params = []
        for s in sizes:
            for d in dists:
                imtemp = nlm(img, size=s, dist=d)
                e = peak_signal_noise_ratio(img, imtemp)
                img_params.append(imtemp)
                params.append({'size': s, 'dist': d, 'psnr': e})
        return img_params, params 

    def plot_parameters_imgs(self, title,img, params):
        assert len(img) == len(params)

        n = len(img)
        first_params_len = len(set(obj[next(iter(obj))] for obj in params))
        if len(params[0]) > 2:
            rows = first_params_len
            print(rows)
        else:
            rows = n // 5 + (n % 5 > 0)
        cols = len(params) // rows + (len(params) % rows > 0)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        axs = axs.ravel()  # Flatten the array of axes

        for i in range(n):
            axs[i].imshow(img[i], cmap='gray')
            axs[i].set_xlabel('\n'.join(f'{key}: {value:.3f}' for key, value in params[i].items()))
            axs[i].set_xticks([])  # Disable x-axis ticks
            axs[i].set_yticks([])  # Disable y-axis ticks

        # Hide unused subplots
        if n < len(axs):
            for i in range(n, len(axs)):
                axs[i].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, bottom=0.1, right=0.8)  # Increase the vertical space between subplots
        #save img as png
        plt.savefig(f'{title}_plot.png')
        plt.show()

    def launch_protocol(self, img):
        #Mode of launch
        print('Launch parameters estimator ? y/N')
        if input() == 'y':
            print('Launch parameters estimator...')
            self.parameters.parameters_estimator(img)
            self.update_parameters()

    def gaussian(self, img):
        return nd.gaussian_filter(img,sigma=self.sigma)

    def bilat(self, img):
        g = estimate_sigma(img, average_sigmas=True)
        return denoise_bilateral(img, sigma_spatial=g)

    def wavelett(self, img, mode=1):  #MODE 1 == VISU, MODE 2 == BAYES
        if mode==1:
            return denoise_wavelet(img, method='VisuShrink', mode='soft', rescale_sigma=True)
        return denoise_wavelet(img, method='BayesShrink', mode='soft', rescale_sigma=True)

    def wavelett_bayes(self, img):
        ret = self.wavelett(img,mode=2)
        ret -= ret.min()
        ret /= ret.max() 
        return ret

    def wavelett_visu(self, img):
        return self.wavelett(img,mode=1)

    def totvar(self, img):
        return denoise_tv_chambolle(img, weight=self.w)

    def aniso(self, img):
        return anisotropic_diffusion(img, niter=self.n_it, kappa=self.kappa, gamma=self.gamma, option=2)

    def nlm(self, img):
        g = estimate_sigma(img, average_sigmas=True)
        return denoise_nl_means(img, h=1.15 * g, fast_mode=True,patch_size=self.size, patch_distance=self.dist)

