import os
import numpy as np
from skimage import img_as_ubyte, io
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def reverse_normalize(img):
    return img * (np.max(img) - np.min(img)) + np.min(img)

def plot_imgs(*img):
    fig, ax = plt.subplots(1, len(img), figsize=(15, 15))
    if len(img) == 1:
        ax.imshow(img[0], cmap='gray')
        ax.axis('off')
    else:
        for i in range(len(img)):
            ax[i].imshow(img[i], cmap='gray')
            ax[i].axis('off')
    plt.show()

def check_psnr(img, cleaned_img, title):
    cleaned_psnr = peak_signal_noise_ratio(img, cleaned_img)
    print(f"PSNR with {title}: {cleaned_psnr}")

def save_img_as_png(img, result_dir, filename):
    os.makedirs(result_dir, exist_ok=True)
    img = np.squeeze(img)
    if len(img.shape) == 2:
        filename = os.path.join(result_dir, filename)
        io.imsave(filename, img_as_ubyte(img), check_contrast=False)
