import filt.filters as flt
import numpy as np
from skimage import (img_as_float,io)
from matplotlib import pyplot as plt
import sys
if __name__== '__main__':
    print('Noise image and calculate Peak SNR ? Y/n')
    b=input()
    if b=='n' or b=='N' or b=='no' or b=='No':
        b=False
    elif b=='Y' or b=='y' or b=='yes' or b=='Yes' or not b:
        b=True
    else :
        print('Not a valid answer')
        sys.exit()

    
    inpath="/home/tpx1/Bureau/algo/test.nii"
    outpath="/home/tpx1/Bureau/algo/"
    img=img_as_float(io.imread(inpath))[0]    
    t=flt.gaussian(img,outpath)
    plt.figure
    plt.imshow(t,cmap='gray')
    plt.show()
