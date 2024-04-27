import numpy as np
from skimage import (img_as_float, img_as_ubyte ,io)
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import resize
from multiprocessing import Pool
from skimage.util import random_noise
import nibabel as nib
import csv
from itertools import permutations
from MIMIQ import filters as flt
from MIMIQ.utils import normalize, save_img_as_png

def apply_filter_permutation(i, filter_order, img):
    print(f"\rApplying filter permutation {i+1}...", flush=True, end='')
    temp_img = img.copy()
    results = []
    for j, filter_func in enumerate(filter_order):
        temp_img = filter_func(temp_img)
        filename = {f'{k+1}': filter_order[k].__name__ if k <= j else None for k in range(len(filter_order))}
        filename = '_'.join([f"{v}" for k, v in filename.items() if v is not None])
        filename = f"{filename}.png"
        save_img_as_png(temp_img, "img_result/", filename)
        if temp_img.shape != img.shape:
            temp_img = resize(temp_img, img.shape)
        psnr = peak_signal_noise_ratio(img, temp_img)
        result = {f'filter_{k+1}': filter_order[k].__name__ if k <= j else None for k in range(len(filter_order))}
        result['psnr'] = psnr
        result['img_link'] = "img_result/" + filename
        results.append(result)
        results.append(result)
    return results, temp_img

class ImageFiltering:
    def __init__(self, inpath=None, img_num=0):
        self.inpath = inpath
        self.image_num = img_num
        self.filt = None
        self.original_image = nib.load(self.inpath).get_fdata()[:,:,self.image_num]
        self.original_image = np.squeeze(self.original_image)
        self.normalized_image = normalize(self.original_image).astype(np.float32)

    def process_all_filters(self):   
        def apply_filters(filters, img):
            print("Applying filters...", flush=True, end='')
            with Pool() as p:
                args = [(i, filter_order, img) for i, filter_order in enumerate(permutations(filters))]
                results = p.starmap(apply_filter_permutation, args)
            with open('psnrs.csv', 'w', newline='') as f:
                fieldnames = [f'filter_{i+1}' for i in range(len(filters))] + ['psnr', 'resemblance', 'img_link']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                written_rows = set()
                for result in results:
                    for row in result[0]:
                        row_tuple = tuple(row.items())
                        if row_tuple not in written_rows:
                            writer.writerow(row)
                            written_rows.add(row_tuple)
            return [result[1] for result in results]
        
        self.filt = flt.Filters()
        self.filt.launch_protocol(self.normalized_image)     

        return apply_filters(self.filt.filters, self.normalized_image)