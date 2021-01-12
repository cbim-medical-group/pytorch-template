"""
subsample slices and shift the short axis and long axis slices
"""
import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift


class ExpSaLaSegSelection:
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

    def __init__(self, sa_slices_num=13, la_views_num=3, perturbation=False, hist_clamp=False):
        self.sa_slices_num = sa_slices_num
        self.la_views_num = la_views_num
        self.perturbation = perturbation
        self.hist_clamp_enable = hist_clamp

    def __call__(self, sample):
        # Here we just use mask instead of image as input.
        # Be careful if you use this script otherwise!
        image, mask, misc = sample['image'],sample['mask'], sample['misc']

        sax_image = image[0:1]
        lax_image = image[1:]
        # Select Short Axis slices
        sax_mask = mask[0:1]
        lax_mask = mask[1:]
        laxch, sax_num, x, y = lax_mask.shape
        if sax_num < self.sa_slices_num:
            sax_mask = np.pad(sax_mask, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))
            lax_mask = np.pad(lax_mask, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))

            sax_image = np.pad(sax_image, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))
            lax_image = np.pad(lax_image, ((0, 0), (0, self.sa_slices_num - sax_num), (0, 0), (0, 0)))


        elif sax_num > self.sa_slices_num:
            start_idx = np.random.randint(0, sax_num - self.sa_slices_num)
            sax_mask = sax_mask[:, start_idx:(start_idx + self.sa_slices_num)]
            lax_mask = lax_mask[:, start_idx:(start_idx + self.sa_slices_num)]

            sax_image = sax_image[:, start_idx:(start_idx + self.sa_slices_num)]
            lax_image = lax_image[:, start_idx:(start_idx + self.sa_slices_num)]

        # Select Long views
        lax_list = list(range(0, laxch))
        if laxch >= self.la_views_num:
            lax_indexes = sorted(random.sample(lax_list, k=self.la_views_num))
            lax_mask = lax_mask[lax_indexes]
            lax_image = lax_image[lax_indexes]
        else:
            lax_mask = np.pad(lax_mask, ((0, self.la_views_num - laxch), (0,0), (0, 0), (0, 0)))
            lax_image = np.pad(lax_image, ((0, self.la_views_num - laxch), (0,0), (0, 0), (0, 0)))


        mask = np.concatenate((sax_mask, lax_mask), axis=0).astype("float")
        image = np.concatenate((sax_image, lax_image), axis=0).astype("float")

        if self.hist_clamp_enable:
            image = self.hist_clamp(image)

        return {'image': image, 'mask': mask, 'misc': misc}

    def hist_clamp(self, image):
        new_image = []
        bd_up = 99.9
        bd_low = 0.1
        for i in range(image.shape[0]):
            sub_img = image[i]
            sub_img[sub_img > np.percentile(sub_img, bd_up)] = np.percentile(sub_img, bd_up)
            sub_img[sub_img < np.percentile(sub_img, bd_low)] = np.percentile(sub_img, bd_low)
            new_image.append(sub_img)
        new_image = np.stack(new_image, axis=0)
        return new_image



