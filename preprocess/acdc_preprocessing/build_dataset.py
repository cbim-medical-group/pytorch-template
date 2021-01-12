import h5py
import numpy as np
import os
import scipy.ndimage as ndimage

from preprocess.acdc_preprocessing.acdc_roi_generation import calc_rois


def get_acdc_datasets(path, type):
    file = h5py.File(path, 'r')
    case_list = list(file[f"{type}"])
    # case_list = case_list[:10]
    return_ds = []
    for case in case_list:
        volume_list = list(file[f"{type}/{case}"])
        volume_4d = file[f"{type}/{case}/data_without_label"][()]

        for volume_id in volume_list:
            if volume_id != "data_without_label":
                idx = int(volume_id) - 1
                annotated_volume = file[f"{type}/{case}/{volume_id}/data"][()]
                annotated_mask = file[f"{type}/{case}/{volume_id}/label"][()]
                input = {"idx": idx, "annotated_volume": annotated_volume, "annotated_mask": annotated_mask,
                         "volume_4d": volume_4d, "path": f"{type}/{case}/{volume_id}",
                         "x_spacing": file[f"{type}/{case}/{volume_id}/data"].attrs['x_spacing'],
                         "y_spacing": file[f"{type}/{case}/{volume_id}/data"].attrs['y_spacing'],
                         "z_spacing": file[f"{type}/{case}/{volume_id}/data"].attrs['z_spacing']}
                return_ds.append(input)
    return return_ds


def calc_bbox(rois, h, w, padding=10):
    coords = np.where(rois > 0)

    result = [max(min(coords[1]) - padding, 0), min(max(coords[1]) + padding, h), max(min(coords[2]) - padding, 0),
              min(max(coords[2]) + padding, w)]
    print(f"bbox: {result}")
    return result


# def calc_bbox(circles, h, w):
#     padding = 10
#     bbox = [h, 0, w, 0]
#     for i in circles:
#         bbox[0] = min(bbox[0], i[0][0] - 2 * i[1] - padding)
#         bbox[1] = max(bbox[1], i[0][0] + padding)
#         bbox[2] = min(bbox[2], i[0][1] - padding)
#         bbox[3] = max(bbox[3], i[0][1] + 2 * i[1] + padding)
#     print(bbox)
#     return bbox


def start_process(path, target, type="train", t=1, s=1, spacing=None):
    dataset = get_acdc_datasets(path, type)
    final_dataset_sax = []
    final_dataset_lax_1 = []
    final_dataset_lax_2 = []
    sum_ref_4d = 0
    for ds in dataset:
        idx = ds['idx']
        path = ds['path']
        path = path.replace("/", "-")
        print(f"Start path:{path}")

        volume = ds['annotated_volume']
        ref_4d = ds['volume_4d']
        mask = ds['annotated_mask']



        volume = np.moveaxis(volume, -1, 0)
        mask = np.moveaxis(mask, -1, 0)

        ref_4d = np.moveaxis(ref_4d, 2, 0)
        ref_4d = np.moveaxis(ref_4d, -1, 1)
        s_num, t_num, h, w = ref_4d.shape

        if sum_ref_4d != np.sum(ref_4d):
            rois, circles = calc_rois(ref_4d)
            sum_ref_4d = np.sum(ref_4d)

        bbox = calc_bbox(rois, h, w, padding=15)
        bbox = [int(round(j)) for j in bbox]

        # roi_mask = np.zeros((mask.shape[0],bbox[1]-bbox[0], bbox[3]-bbox[2]))
        roi_mask = mask[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]
        # roi_volume = np.zeros((volume.shape[0], bbox[1]-bbox[0], bbox[3]-bbox[2]))
        roi_volume = volume[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]
        roi_ref_4d = ref_4d[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3]]

        if np.sum(roi_mask) != np.sum(mask):
            print(f"ROI is good enough? | {np.sum(roi_mask)} vs. {np.sum(mask)}")

        spacing_id = ""
        if spacing is not None:
            x_spacing = ds['x_spacing']  # 1.4, 0.6mm spacing
            zoom_rate = x_spacing / spacing
            print(f"from: {x_spacing} mm, to: {spacing}mm...")
            roi_volume = ndimage.zoom(roi_volume, zoom=(1, zoom_rate, zoom_rate), order=0)
            roi_ref_4d = ndimage.zoom(roi_ref_4d, zoom=(1, 1, zoom_rate, zoom_rate), order=0)
            roi_mask = ndimage.zoom(roi_mask, zoom=(1, zoom_rate, zoom_rate), order=0)
            bbox = [int(round(j*zoom_rate)) for j in bbox]
            spacing_id = f'_iso{spacing}'

        for slice_id in range(volume.shape[0]):
            slice = np.zeros((5, roi_volume.shape[1], roi_volume.shape[2]))
            # -t and +t slice as channels
            t_index_1 = (idx - t + t_num) % t_num
            t_index_2 = (idx + t + t_num) % t_num
            slice[0] = roi_ref_4d[slice_id, t_index_1, :, :]
            slice[1] = roi_ref_4d[slice_id, t_index_2, :, :]
            # current slice as channel
            slice[2] = roi_volume[slice_id, :, :]

            # -s and +s slice as channels
            s_index_1 = slice_id - s
            s_index_2 = slice_id + s
            if s_index_1 >= 0:
                slice[3] = roi_volume[s_index_1, :, :]
            if s_index_2 < roi_volume.shape[0]:
                slice[4] = roi_volume[s_index_2, :, :]

            roi_mask_sax = roi_mask[slice_id, :, :]

            final_dataset_sax.append(
                {'idx': idx, 'orientation': 'sax', 'bbox': bbox, 'slice_id': slice_id, 'slice': slice,
                 'mask': roi_mask_sax, 'path': path})

        print(f"Finish Sax for path:{path}")

        # LAX 1
        for slice_id in range(roi_volume.shape[1]):
            slice = np.zeros((5, roi_volume.shape[0], roi_volume.shape[2]))
            # -t and +t slice as channels
            t_index_1 = (idx - t + t_num) % t_num
            t_index_2 = (idx + t + t_num) % t_num
            slice[0] = roi_ref_4d[:, t_index_1, slice_id, :]
            slice[1] = roi_ref_4d[:, t_index_2, slice_id, :]

            # current slice as channel
            slice[2] = roi_volume[:, slice_id, :]

            # -s and +s slice as channels
            s_index_1 = slice_id - 1
            s_index_2 = slice_id + 1
            if s_index_1 >= 0:
                slice[3] = roi_volume[:, s_index_1, :]
            if s_index_2 < roi_volume.shape[1]:
                slice[4] = roi_volume[:, s_index_2, :]

            roi_mask_lax1 = roi_mask[:, slice_id, :]

            final_dataset_lax_1.append(
                {'idx': idx, 'orientation': 'lax1', 'bbox': bbox, 'slice_id': slice_id, 'slice': slice,
                 'mask': roi_mask_lax1, 'path': path})
        print(f"Finish Lax 1 for path:{path}")

        # LAX 2
        for slice_id in range(roi_volume.shape[2]):
            slice = np.zeros((5, roi_volume.shape[0], roi_volume.shape[1]))
            # -t and +t slice as channels
            t_index_1 = (idx - t + t_num) % t_num
            t_index_2 = (idx + t + t_num) % t_num
            slice[0] = roi_ref_4d[:, t_index_1, :, slice_id]
            slice[1] = roi_ref_4d[:, t_index_2, :, slice_id]

            # current slice as channel
            slice[2] = roi_volume[:, :, slice_id]

            # -s and +s slice as channels
            s_index_1 = slice_id - 1
            s_index_2 = slice_id + 1
            if s_index_1 >= 0:
                slice[3] = roi_volume[:, :, s_index_1]
            if s_index_2 < roi_volume.shape[2]:
                slice[4] = roi_volume[:, :, s_index_2]

            roi_mask_lax2 = roi_mask[:, :, slice_id]

            final_dataset_lax_2.append(
                {'idx': idx, 'orientation': 'lax2', 'bbox': bbox, 'slice_id': slice_id, 'slice': slice,
                 'mask': roi_mask_lax2, 'path': path})

        print(f"Finish Lax 2 for path:{path}")

    if os.path.isfile(os.path.join(target, f"ACDC_Miccai_sax{spacing_id}.h5")):
        os.remove(os.path.join(target, f"ACDC_Miccai_sax{spacing_id}.h5"))
    if os.path.isfile(os.path.join(target, f"ACDC_Miccai_lax1{spacing_id}.h5")):
        os.remove(os.path.join(target, f"ACDC_Miccai_lax1{spacing_id}.h5"))
    if os.path.isfile(os.path.join(target, f"ACDC_Miccai_lax2{spacing_id}.h5")):
        os.remove(os.path.join(target, f"ACDC_Miccai_lax2{spacing_id}.h5"))

    f1 = h5py.File(os.path.join(target, f"ACDC_Miccai_sax{spacing_id}.h5"), 'w')
    for ds in final_dataset_sax:
        f1.create_dataset(f"{ds['path']}-{ds['slice_id']}/data", data=ds['slice'])
        f1.create_dataset(f"{ds['path']}-{ds['slice_id']}/label", data=ds['mask'])
        f1.create_dataset(f"{ds['path']}-{ds['slice_id']}/bbox", data=ds['bbox'])
    f1.close()

    f2 = h5py.File(os.path.join(target, f"ACDC_Miccai_lax1{spacing_id}.h5"), 'w')
    for ds in final_dataset_lax_1:
        f2.create_dataset(f"{ds['path']}-{ds['slice_id']}/data", data=ds['slice'])
        f2.create_dataset(f"{ds['path']}-{ds['slice_id']}/label", data=ds['mask'])
        f2.create_dataset(f"{ds['path']}-{ds['slice_id']}/bbox", data=ds['bbox'])
    f2.close()

    f3 = h5py.File(os.path.join(target, f"ACDC_Miccai_lax2{spacing_id}.h5"), 'w')
    for ds in final_dataset_lax_2:
        f3.create_dataset(f"{ds['path']}-{ds['slice_id']}/data", data=ds['slice'])
        f3.create_dataset(f"{ds['path']}-{ds['slice_id']}/label", data=ds['mask'])
        f3.create_dataset(f"{ds['path']}-{ds['slice_id']}/bbox", data=ds['bbox'])
    f3.close()


# start_process("/share_hd1/projects/code_seg_qi/db/ACDC/processed/ACDC_train.h5",
#               "/share_hd1/projects/code_seg_qi/db/ACDC/processed/", "train", t=3, s=1, spacing=1)

start_process("/share_hd1/projects/code_seg_qi/db/ACDC/processed/ACDC_train.h5",
              "/share_hd1/projects/code_seg_qi/db/ACDC/processed/", "train", t=3, s=1)

