import SimpleITK as sitk
import h5py
import numpy as np
import pydicom
import re
from cv2 import cv2

from utils import getListOfFiles


def create_db_3d(root_path, target_path):
    data_path_list = getListOfFiles(root_path, [".dcm"])
    path_re = "(.*raw)/(DET[^\/]+)/(DET[^_]+)_([^_]+)_([^.]+)\.dcm"

    target = h5py.File(target_path,'w')

    data_info = {}
    for path in data_path_list:
        search_result = re.search(path_re, path, re.IGNORECASE)
        if search_result is None:
            print(f"The path is not qualified,{path}")

        root_path = search_result[1]
        case_id = search_result[2]
        series_id = search_result[4]
        phase_id = search_result[5]

        if case_id not in data_info:
            data_info[case_id] = {}
        if phase_id not in data_info[case_id]:
            data_info[case_id][phase_id] = {"LA": [], "SA": []}

        if "LA" in series_id:
            data_info[case_id][phase_id]["LA"].append((path, int(series_id.replace("LA", ""))))
        else:
            data_info[case_id][phase_id]["SA"].append((path, int(series_id.replace("SA", ""))))

    for case_id in data_info:
        for phase_id in data_info[case_id]:
            print(f"process data:{case_id} - phase:{phase_id}")
            data = data_info[case_id][phase_id]
            la_src = data['LA']
            la_src = sorted(la_src, key=lambda tup: tup[1])
            la_src = [tup[0] for tup in la_src]
            sa_src = data['SA']
            sa_src = sorted(sa_src, key=lambda tup: tup[1])
            sa_src = [tup[0] for tup in sa_src]

            sa_arr = [pydicom.dcmread(sa).pixel_array for sa in sa_src]
            sa_arr_shape = [arr.shape for arr in sa_arr]
            if len(set(sa_arr_shape)) != 1:
                continue
            sa_arr = np.stack(sa_arr)
            sa_mask_arr = np.stack([cv2.imread(sa.replace('dcm','png'), flags=cv2.IMREAD_GRAYSCALE) for sa in sa_src])
            sa_mask_arr[sa_mask_arr>0]=255
            la_all_arr = []
            la_all_mask_arr = []

            for la_path in la_src:
                la_img = sitk.ReadImage(la_path)
                la_mask_img = create_img_from_mask(la_path, la_path)
                la_arr = []
                la_mask_arr = []
                for sa_path in sa_src:
                    sa_img = sitk.ReadImage(sa_path)
                    # sa_mask_img = create_img_from_mask(sa_path, sa_path)
                    la_img_project2sa = interpolate_img_itk(la_img, sa_img)
                    la_mask_project2sa = interpolate_img_itk(la_mask_img, sa_img)
                    la_arr.append(sitk.GetArrayFromImage(la_img_project2sa).squeeze())
                    la_mask_arr.append(sitk.GetArrayFromImage(la_mask_project2sa).squeeze())
                la_all_arr.append(np.stack(la_arr))
                la_all_mask_arr.append(np.stack(la_mask_arr))

            la_all_arr = np.stack(la_all_arr)
            la_all_mask_arr = np.stack(la_all_mask_arr)

            target.create_dataset(f"train/{case_id}_{phase_id}/sa/data",data=sa_arr, compression='gzip')
            target.create_dataset(f"train/{case_id}_{phase_id}/sa/label",data=sa_mask_arr, compression='gzip')
            target.create_dataset(f"train/{case_id}_{phase_id}/la/data",data=la_all_arr, compression='gzip')
            target.create_dataset(f"train/{case_id}_{phase_id}/la/label",data=la_all_mask_arr, compression='gzip')

    target.close()

def create_img_from_mask(mask_src, meta_source_src):
    mask = cv2.imread(mask_src.replace("dcm", "png"), flags=cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    mask = mask[np.newaxis, :, :]
    meta_source_img = sitk.ReadImage(meta_source_src)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetOrigin(meta_source_img.GetOrigin())
    mask_img.SetSpacing(meta_source_img.GetSpacing())
    mask_img.SetDirection(meta_source_img.GetDirection())
    return mask_img


def interpolate_img_itk(itkimg_src, itkimg_dst, order=1):
    if order == 0:
        method = sitk.sitkNearestNeighbor
    elif order == 1:
        method = sitk.sitkLinear
    elif order == 2:
        method = sitk.sitkBSpline

    new_spacing = itkimg_dst.GetSpacing()

    new_origin = itkimg_dst.GetOrigin()

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(itkimg_dst.GetDirection())
    resampler.SetSize(itkimg_dst.GetSize())
    resampler.SetInterpolator(method)

    imgResampled = sitk.Cast(resampler.Execute(sitk.Cast(itkimg_src, sitk.sitkFloat32)), itkimg_dst.GetPixelIDValue())

    return imgResampled


create_db_3d("/research/cbim/vast/qc58/pub-db/CAP/raw", "/research/cbim/vast/qc58/pub-db/CAP/process/MC-Net/processed_data.h5")
