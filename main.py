import MIMIQ.filters as flt
import numpy as np
import sys
from MIMIQ.imageFiltering import ImageFiltering

if __name__== '__main__':
    inpath = sys.argv[1]       #Path of image
    img_num = int(sys.argv[2])        #Image order in NII file
    img = ImageFiltering(inpath, img_num)
    img.process_all_filters()
    print("Process finished.")