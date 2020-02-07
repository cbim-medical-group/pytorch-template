import cv2
import h5py
import numpy as np
import scipy.interpolate
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

def smooth_obj(from_path, to_path):
    from_h5_file = h5py.File(from_path, 'r')
    to_h5_file = h5py.File(to_path, 'w')
    from_list = list(from_h5_file)

    for series_id in from_list:
        print(f"process...{series_id}")
        orig_array = from_h5_file[series_id + "/label"][()]
        print(np.unique(orig_array))
        denoised_array = smooth_obj_one_case(orig_array, scale=4)
        to_h5_file.create_dataset(series_id + "/label", data=denoised_array)

    from_h5_file.close()
    to_h5_file.close()


def smooth_obj_one_case(orig_array, scale=2):
    scale_array = []
    for i in range(orig_array.shape[-1]):
        slice = orig_array[:, :, i]
        scale_slice = resize(slice, (slice.shape[0] * scale, slice.shape[1] * scale), order=0,
                             preserve_range=True)
        final_mask = np.zeros_like(scale_slice)
        unique_id = np.unique(scale_slice)[1:].tolist()
        if len(unique_id) == 0:
            scale_array.append(scale_slice)
            continue
        for label_id in unique_id:
            show_img = np.zeros_like(scale_slice)
            show_img[scale_slice == label_id] = 1

            blurred_img = gaussian_filter(show_img, sigma=4)
            blurred_img[blurred_img > 0.5] = 1
            blurred_img[blurred_img <= 0.5] = 0
            final_mask[blurred_img == 1] = label_id
        scale_array.append(final_mask)

    return np.stack(scale_array, axis=-1)


smooth_obj(
    "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-mask-postprocess.h5",
    "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-final-x4.h5")
