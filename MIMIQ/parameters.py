# parameters.py
from skimage.util import random_noise
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,denoise_wavelet, estimate_sigma,denoise_nl_means)
from skimage.metrics import peak_signal_noise_ratio
from medpy.filter.smoothing import anisotropic_diffusion
from scipy import ndimage as nd
from MIMIQ.utils import save_img_as_png
import numpy as np
import warnings
import sys

class Parameters:
    def __init__(self):
        self.NLM_size = 9
        self.NLM_dist = 5
        self.aniso_iter = 25
        self.aniso_kappa = 50
        self.aniso_gamma = 0.02
        self.totvar_weight = 0.1
        self.gaussian_sigma = 1
        self.bestsaved = False

    def set_best_saved(self, value, g_sigma, t_weight, a_iter, a_kappa, a_gamma, N_size, N_dist):
        self.bestsaved = value
        self.gaussian_sigma = g_sigma
        self.totvar_weight = t_weight
        self.aniso_iter = a_iter
        self.aniso_kappa = a_kappa
        self.aniso_gamma = a_gamma
        self.NLM_size = N_size
        self.NLM_dist = N_dist


    def parameters_estimator(self, img):
        noisy_img = random_noise(img, mode='Gaussian', rng=None, clip=True).astype(np.float32)
        #GAUSSIAN ESTIMATOR
        save_img_as_png(img, "img_result/", 'Base.png')
        save_img_as_png(noisy_img, "img_result/", 'Noisy.png')
        best_sig=0
        best_snr_g=peak_signal_noise_ratio(img, noisy_img)
        sigmas = np.linspace(0.1,2,1000)

        print('Estimating best Gaussian parameters...', end='', flush=True)
        for i, sig in enumerate(sigmas):
            imtemp = nd.gaussian_filter(noisy_img, sigma=sig)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e=peak_signal_noise_ratio(img, imtemp)
            if e==float('inf'):
                continue
            if e>best_snr_g:
                best_snr_g=e
                best_sig=sig
        print('\rGaussian parameter estimation done.          ')

        #TOT VAR ESTIMATOR
        best_est=0.1
        best_snr_t=peak_signal_noise_ratio(img, noisy_img)
        print('Estimating best Total Variation parameters...', end='', flush=True)
        for i in np.linspace(0.1,1,100):
            print('\rEstimating best Total Variation parameters... w: {:.3f}'.format(i), end='', flush=True)
            imtemp = denoise_tv_chambolle(noisy_img, weight=i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e=peak_signal_noise_ratio(img, imtemp)
            if e>best_snr_t:
                best_snr_t=e
                best_est=i
        print('\rTotal Variation parameter estimation done.                  ')


        #ANISOTROPIC ESTIMATOR

        best_n=0
        best_g=0
        best_k=0
        best_snr_a=peak_signal_noise_ratio(img, noisy_img)
        print('Estimating best Anisotropic parameters...', end='', flush=True)
        for n in range(1,10):
            for g in np.linspace(0.01,0.25,25):
                for k in range(20,80):
                    print('\rEstimating best Anisotropic parameters... n: {} g: {:.3f} k: {:.3f}               '.format(n, g, k), end='', flush=True)
                    imtemp=anisotropic_diffusion(noisy_img, niter=n, kappa=k, gamma=g)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        e=peak_signal_noise_ratio(img, imtemp)
                    if e>best_snr_a:
                        best_snr_a=e
                        best_n=n
                        best_g=g
                        best_k=k
        print('\rAnisotropic parameter estimation done.                                 ')



        #NLM ESTIMATOR

        best_size,best_dist=0,0
        best_snr_n=peak_signal_noise_ratio(img, noisy_img)
        print('Estimating best NLM parameters...', end='', flush=True)
        for s in range(1,10):
            for d in range(1,10):
                print('\rEstimating best NLM parameters... size: {} dist: {}'.format(s, d), end='', flush=True)
                imtemp = denoise_nl_means(noisy_img, h=0.8*estimate_sigma(noisy_img, average_sigmas=True), fast_mode=True, patch_size=s, patch_distance=d)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    e=peak_signal_noise_ratio(img, imtemp)
                if e>best_snr_n:
                    best_snr_n=e
                    best_size,best_dist=s,d
        print('\rNLM parameter estimation done.                                      ')

        #PRINTING
        print('Base PSNR: {:.5f}'.format(peak_signal_noise_ratio(img, noisy_img)))
        print('Best Gaussian sigma: {:.3f}'.format(best_sig))
        print('Best Total Variation weight: {:.3f}'.format(best_est))
        print('Best Anisotropic parameters: n: {} g: {:.3f} k: {:.3f}'.format(best_n,best_g,best_k))
        print('Best NLM parameters: size: {} dist: {}'.format(best_size,best_dist))
        self.set_best_saved(True, best_sig, best_est, best_n, best_k, best_g, best_size, best_dist)
