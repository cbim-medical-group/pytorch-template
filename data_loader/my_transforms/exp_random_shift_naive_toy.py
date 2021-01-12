"""
subsample slices and shift the short axis and long axis slices
"""
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift

from preprocess.MMWHS.create_dataset_FIMH import subsample_label


class ExpRandomShiftNaive:
    """
    Subsample the 3D numpy [h, w, d] by slice max number.
    """

    def __init__(self, random_shift_rate=0.8, random_shift_range=10, sample_number=10, lx_perturbation=True, fix_d=10):
        self.perturbation = perturbation

        self.sample_number = sample_number
        self.channel_number = 3
        self.random_shift_rate = random_shift_rate
        self.random_shift_range = random_shift_range
        self.lx_shift_enable = lx_perturbation
        self.fix_d = fix_d

    def move_mass_to_center(self, mask):
        h, w, d = mask.shape
        ch, cw, cd = ndimage.measurements.center_of_mass(mask)
        mask = np.roll(mask, (int(round(h // 2 - ch)), int(round(w // 2 - cw))), axis=(0, 1))
        return mask

    def __call__(self, sample):
        # Here we just use mask instead of image as input. Be careful if you use this script otherwise!
        mask, misc = sample['data'], sample['misc']

        image, orig_img, sax_shift, lax_shift = subsample_label(mask, self.random_shift_rate, self.random_shift_range,
                                                                self.sample_number, self.lx_shift_enable, True)



        # offset_arr = np.array(offset_arr) / (119 / 2)
        offset_arr = np.array(offset_arr) / 10

        # offset_arr = offset_arr * 3

        return {'image': image, 'mask': np.array(offset_arr), 'misc': misc}
