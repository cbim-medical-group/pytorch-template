
import os
import numpy as np
import SimpleITK as sitk
import sys

from cv2 import cv2


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

def extract_MPR(droot, src_list, dst_list, tmp_dir = 'temp', order=1, suffix='.nii'):
    orig_tmp_dir = tmp_dir
    tmp_dir = os.path.join(droot, tmp_dir)
    
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    if not os.path.isdir(orig_tmp_dir):
        os.mkdir(orig_tmp_dir)
    # src:la, dst:sa
    for case in src_list:
        # continue
        case_file = os.path.join(droot, case)
        slice_src = sitk.ReadImage(case_file)

        mask = cv2.imread(case_file.replace("dcm", "png"), flags=cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255
        mask = mask[np.newaxis, :, :]
        mask_src = sitk.GetImageFromArray(mask)
        mask_src.SetOrigin(slice_src.GetOrigin())
        mask_src.SetSpacing(slice_src.GetSpacing())
        mask_src.SetDirection(slice_src.GetDirection())

        src_to_dst_list = []
        for ref_case in dst_list:
            ref_file = os.path.join(droot, ref_case)

            slice_dst = sitk.ReadImage(ref_file)
            
            slice_resampled = interpolate_img_itk(slice_src, slice_dst, order=order)
            mask_resampled = interpolate_img_itk(mask_src, slice_dst, order=order)

            new_name = case[:-4]+'_2_'+ref_case[:-4]
            np_src = norm_img(sitk.GetArrayFromImage(slice_src))
            np_dst = norm_img(sitk.GetArrayFromImage(slice_dst))
            np_resample = norm_img(sitk.GetArrayFromImage(slice_resampled))
            np_resample_mask = norm_img(sitk.GetArrayFromImage(mask_resampled))

            cv2.imwrite(os.path.join(orig_tmp_dir, new_name+"_la.png"),np_src)
            cv2.imwrite(os.path.join(orig_tmp_dir, new_name+"_sa.png"),np_dst)
            cv2.imwrite(os.path.join(orig_tmp_dir, new_name+"_resampled.png"),np_resample)
            cv2.imwrite(os.path.join(orig_tmp_dir, new_name+"_resampled_mask.png"),np_resample_mask)
            # sitk.WriteImage(slice_resampled, os.path.join(orig_tmp_dir, new_name+suffix))
        

def norm_img(src):
    src = src.squeeze()
    return ((src-src.min())/src.max())*255


if __name__ == "__main__":
    ## CAP data
    droot = r'/research/cbim/vast/qc58/pub-db/CAP/raw/DET0006301'
    
    dcm_list = os.listdir(droot)
    dcm_list = [file for file in dcm_list if file[-3:] == 'dcm']
    dcm_list_ph0 = [file for file in dcm_list if '_ph0' in file]
    
    la_list = [file for file in dcm_list_ph0 if '_LA' in file]
    sa_list = [file for file in dcm_list_ph0 if '_SA' in file]
    
    extract_MPR(droot, la_list, sa_list, tmp_dir = 'MPR_lax2sax', order=0)

