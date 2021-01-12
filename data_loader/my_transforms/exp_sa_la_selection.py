"""
subsample slices and shift the short axis and long axis slices
"""
import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift


class ExpSaLaSelection:
    """
    Select SA {sa_slices_num} slices as 1 x {sa_slices_num} x X x Y and select LA {la_views_num} views as {la_views_num} x {sa_slices_num} x X x Y.
    combine as {1 + la_views_num} x {sa_slices_num} x X x Y image and label.
    X:256, Y:256

    # la number: [1, 2, 3, 4, 5, 6, 7] : [ 25, 105, 999, 643, 215, 184,  45]
    # sa slice numbers: [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24] : [ 75,  73, 243, 398, 496, 227, 390, 135,  95,  44,  20,  20]
    # x: [192, 240, 256, 288, 320, 512] : [ 555,   20, 1421,   70,   50,  100]
    # y: [156, 162, 174, 186, 192, 200, 208, 216, 224, 240, 256, 288, 320,
    #         512] : [ 200,   50,   50,   25,  294,   85,  117,   40,   40,   20, 1075,
    #           70,   50,  100]
    """

    def __init__(self, sa_slices_num=13, la_views_num=3, perturbation=False):
        self.sa_slices_num = sa_slices_num
        self.la_views_num = la_views_num
        self.perturbation = perturbation

    def __call__(self, sample):
        # Here we just use mask instead of image as input.
        # Be careful if you use this script otherwise!
        mask, misc = sample['mask'], sample['misc']

        # Select Short Axis slices
        sax_mask = mask[0:1]
        lax_mask = mask[1:]
        laxch, sax_num, x, y = lax_mask.shape
        if sax_num < self.sa_slices_num:
            sax_mask = np.pad(sax_mask, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))
            lax_mask = np.pad(lax_mask, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))
        elif sax_num > self.sa_slices_num:
            start_idx = np.random.randint(0, sax_num - self.sa_slices_num)
            sax_mask = sax_mask[:, start_idx:(start_idx + self.sa_slices_num)]
            lax_mask = lax_mask[:, start_idx:(start_idx + self.sa_slices_num)]

        # Select Long views
        lax_list = list(range(0, laxch))
        if laxch >= self.la_views_num:
            lax_indexes = sorted(random.sample(lax_list, k=self.la_views_num))
            lax_mask = lax_mask[lax_indexes]
        else:
            lax_mask = np.pad(lax_mask, ((0, self.la_views_num - laxch), (0,0), (0, 0), (0, 0)))


        mask = np.concatenate((sax_mask, lax_mask), axis=0)

        return {'image': mask, 'mask': mask, 'misc': misc}
