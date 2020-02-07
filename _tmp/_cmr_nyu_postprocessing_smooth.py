import cv2
import h5py
import numpy as np
import scipy.interpolate
from skimage.transform import resize


def smooth_obj(from_path, to_path):
    from_h5_file = h5py.File(from_path, 'r')
    to_h5_file = h5py.File(to_path, 'w')

    for series_id in from_h5_file:
        print(f"process:{series_id}")
        orig_array = from_h5_file[series_id + "/label"][()]
        print(np.unique(orig_array))
        denoised_array = smooth_obj_one_case(orig_array)
        to_h5_file.create_dataset(series_id + "/label", data=denoised_array)

    from_h5_file.close()
    to_h5_file.close()


def smooth_obj_one_case(orig_array, scale=3):
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
            show_img_3ch = np.stack((show_img, show_img, show_img), axis=-1)
            cv_grayimg = cv2.cvtColor(show_img_3ch.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(cv_grayimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            inter_contours_list = []
            for i in contours:
                bc = i.squeeze()
                x = bc[:, 0]
                y = bc[:, 1]
                #     a = np.concatenate((x, [x[0]]))
                #     b = np.concatenate((y, [y[0]]))

                dist = np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)
                dist_along = np.concatenate(([0], dist.cumsum()))

                # build a spline representation of the contour
                spline, u = scipy.interpolate.splprep([x, y], u=dist_along, s=0)

                # resample it at smaller distance intervals
                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                interp_x, interp_y = scipy.interpolate.splev(interp_d, spline)
                #     plot(interp_x, interp_y, '-')
                inter_contours = np.stack((interp_x, interp_y), axis=1)
                inter_contours_list.append(inter_contours.astype(int))

            img_1 = np.zeros_like(show_img_3ch).astype(np.uint8)
            img_1.fill(0)
            smooth_mask_3ch = cv2.drawContours(img_1, inter_contours_list, -1, (255, 0, 0), cv2.FILLED)
            smooth_mask = smooth_mask_3ch[:,:,0]
            final_mask[smooth_mask == 255] = label_id
            scale_array.append(final_mask)

    return np.stack(scale_array, axis=-1)


smooth_obj(
    "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-mask-postprocess.h5",
    "/Users/qichang/PycharmProjects/pytorch-template/data/ACDC/processed/Export-1-Cardial_MRI_DB-0-predict-mask-smooth.h5")
