import cv2
import h5py
import numpy as np
import re
import scipy.interpolate
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

def smooth_obj(from_path, to_path, resize_type ="scale",scale=4, ref_path=None):
    from_h5_file = h5py.File(from_path, 'r')
    to_h5_file = h5py.File(to_path, 'w')
    from_list = list(from_h5_file['train'])
    if ref_path is not None:
        ref_h5_file = h5py.File(ref_path, 'r')


    for series_id in from_list:
        if ref_path is not None:
            desc = ref_h5_file[f'train/{series_id}/data'].attrs['description']
            result = re.findall(r"sax", desc, re.IGNORECASE)
            if result is None:
                continue;


        print(f"process...{series_id}")
        orig_array = from_h5_file["train/"+series_id + "/label"][()]
        print(np.unique(orig_array))
        denoised_array = smooth_obj_one_case(orig_array, resize_type=resize_type, scale=scale)
        to_h5_file.create_dataset("train/"+series_id + "/label", data=denoised_array, compression="gzip")

    from_h5_file.close()
    to_h5_file.close()

def smooth_obj_one_case(orig_array, resize_type="scale", scale=2):
    scale_array = []
    for i in range(orig_array.shape[-1]):
        slice = orig_array[:, :, i]
        if resize_type == "scale" and scale>1:
            slice = resize(slice, (slice.shape[0] * scale, slice.shape[1] * scale), order=0,
                                 preserve_range=True)
        elif resize_type == "size":
            slice = resize(slice, scale, order=0,
                                 preserve_range=True)
        final_mask = np.zeros_like(slice)
        unique_id = np.unique(slice)[1:].tolist()
        if len(unique_id) == 0:
            scale_array.append(slice)
            continue
        for label_id in unique_id:
            show_img = np.zeros_like(slice)
            show_img[slice == label_id] = 1

            blurred_img = gaussian_filter(show_img, sigma=4)
            blurred_img[blurred_img > 0.5] = 1
            blurred_img[blurred_img <= 0.5] = 0
            final_mask[blurred_img == 1] = label_id
        scale_array.append(final_mask)

    return np.stack(scale_array, axis=-1)


# smooth_obj(
#     "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-mask-postprocess.h5",
#     "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-final-x4.h5")

smooth_obj("/ajax/users/qc58/work/projects/pytorch-template/saved/test_results/ACDC_segmentation_exp1/test_result-ACDC_segmentation_exp1.h5","/ajax/users/qc58/work/projects/pytorch-template/saved/test_results/ACDC_segmentation_exp1/test_result-ACDC_segmentation_exp1_512x512_smooth.h5","size",(512,512))
