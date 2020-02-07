"""
This script is for NYU CMR dataset postprocessing step:
1. Will remove small objects prediction, and just keep the largest object for each label, each slice.
2. Will rescale the image as 4 times larger, and smooth the image, and we could check if it's smooth enough or not.

"""
import cv2
import h5py
import numpy as np
import scipy.interpolate
from skimage.transform import resize

from utils.clean_noise import CleanNoise


def remove_small_obj_one_case(orig_array):
    """
    Remove small objects for each label type and each slice.
    :param orig_array:
    :param keep_num:
    :return:
    """
    clean = CleanNoise(top_num=1)
    for i in range(orig_array.shape[-1]):
        slice = orig_array[:, :, i]
        clean_slice = clean.clean_small_obj(slice)
        orig_array[:, :, i] = clean_slice
    return orig_array


def remove_small_obj(from_path, to_path):
    from_h5_file = h5py.File(from_path, 'r')
    to_h5_file = h5py.File(to_path, 'w')
    # to_h5_smooth_file = h5py.File(to_path_smooth, 'w')
    for series_id in from_h5_file:
        print(f"process:{series_id}")
        orig_array = from_h5_file[series_id + "/label"][()]
        denoised_array = remove_small_obj_one_case(orig_array)
        to_h5_file.create_dataset(series_id + "/label", data=denoised_array)

    from_h5_file.close()
    to_h5_file.close()
    # to_h5_smooth_file.close()

remove_small_obj(
    "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-mask-postprocess.h5",
    "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-mask-smooth.h5")


