import h5py
import numpy as np
import os
import random
from scipy import ndimage
from scipy.ndimage.interpolation import shift

from utils import smooth_obj_one_case


def create_dataset(root, file, target, type="train", random_shift_rate=0.4, random_shift_range=5, sample_number=10,
                   lx_shift_enable=False):
    f = h5py.File(os.path.join(root, file), 'r')
    if type == "train":
        db = h5py.File(os.path.join(root, target + "_training" + ".h5"), "w")
        run_list = list(f['train'])[:16]
    elif type == "test":
        db = h5py.File(os.path.join(root, target + "_testing_shift" + str(random_shift_range) + "_lx" + str(
            lx_shift_enable) + ".h5"), "w")
        run_list = list(f['train'])[16:]
    elif type == "validate":
        db = h5py.File(os.path.join(root, target + "_val" + ".h5"), "w")
        run_list = list(f['train'])[16:]

    for idx, i in enumerate(run_list):
        if i == "1008": continue
        label = f[f'train/{i}/label'][()]
        binary_label = np.zeros_like(label)
        binary_label[label == 2] = 1
        # smooth_binary_label = smooth_obj_one_case(binary_label)
        centered_smooth_binary_label = move_mass_to_center(binary_label)

        if type == "train":
            db.create_dataset(f"{type}/{i}/data", data=centered_smooth_binary_label, compression="gzip")
            db.create_dataset(f"{type}/{i}/label", data=centered_smooth_binary_label, compression="gzip")
        elif type =="validate":
            db.create_dataset(f"test/{i}/data", data=centered_smooth_binary_label, compression="gzip")
            db.create_dataset(f"test/{i}/label", data=centered_smooth_binary_label, compression="gzip")
        else:
            subsampled_data, subsampled_label, sax_shift, lax_shift = subsample_label(centered_smooth_binary_label, random_shift_rate,
                                                                random_shift_range, sample_number, lx_shift_enable)

            shifted_data = db.create_dataset(f"{type}/{i}/data", data=subsampled_data, compression="gzip")
            db.create_dataset(f"{type}/{i}/label", data=subsampled_label, compression="gzip")
            shifted_data.attrs['sax_shift'] = sax_shift
            shifted_data.attrs['lax_shift'] = lax_shift

    db.close()
    f.close()

def subsample_label(label, random_shift_rate=0.4, random_shift_range=5, sample_number=10, lx_shift_enable=False, dt_enable=False):
    channel_number = 3
    # mri_img = label[:, :, ::sample_number]
    h, w, d = label.shape
    image = np.zeros((channel_number, h, w, d))
    orig_img = image.copy()

    x, y, z = ndimage.measurements.center_of_mass(label)
    x = int(x)
    y = int(y)
    z = int(z)
    # tip_pos = np.where(label == 1)
    # idx = tip_pos[2].argmin()
    # x = tip_pos[0][idx]
    # y = tip_pos[1][idx]
    # z = tip_pos[2][idx]
    lax_shift = [[0,0],[0,0]]
    if lx_shift_enable:
        r = list(np.random.randint(-random_shift_range, random_shift_range, 4))
        image[1, :, y + r[1], :] = label[:, y, :]
        image[1, :, y + r[1], :] = shift(image[1, :, y + r[1], :], (r[0],0), cval=0, mode='nearest')
        image[2, x + r[2], :, :] = label[x, :, :]
        image[2, x + r[2], :, :] = shift(image[2, x + r[2], :, :], (r[3],0), cval=0, mode='nearest')
        lax_shift = [[r[0], r[1]], [r[2], r[3]]]
    else:
        image[1, :, y, :] = label[:, y, :]
        image[2, x, :, :] = label[x, :, :]
    orig_img[1, :, y, :] = label[:, y, :]
    orig_img[2, x, :, :] = label[x, :, :]

    sax_shift = []
    for i in range(d):
        slice_mri_img = label[:, :, i]
        random_seed = random.random()
        offset_x = offset_y = 0
        if random_seed <= random_shift_rate:
            offset_x = random.randint(-random_shift_range, random_shift_range)
            offset_y = random.randint(-random_shift_range, random_shift_range)
            image[0, :, :, i] = shift(slice_mri_img, (offset_x, offset_y), cval=0, mode='nearest')
            # image[0, :, :, i] = shift(slice_mri_img, (offset_x, offset_y), cval=0, mode='nearest')
            sax_shift.append([offset_x, offset_y])
        else:
            image[0, :, :, i] = slice_mri_img
            sax_shift.append([0,0])
            # image[0, :, :, i] = slice_mri_img
        if dt_enable:
            image[0, :, :, i] = ndimage.distance_transform_edt(1 - image[0, :, :, i])
        orig_img[0, :, :, i] = slice_mri_img

    start_idx = random.randint(0, sample_number - 1)
    image = image[:, :, :, start_idx::sample_number]
    orig_img = orig_img[:, :, :, start_idx::sample_number]
    sax_shift = np.array(sax_shift)[start_idx::sample_number]
    return image, orig_img, list(sax_shift), lax_shift


def move_mass_to_center(mask):
    h, w, d = mask.shape
    ch, cw, cd = ndimage.measurements.center_of_mass(mask)
    mask = np.roll(mask, (int(round(h // 2 - ch)), int(round(w // 2 - cw))), axis=(0, 1))
    return mask


# create_dataset("/research/cbim/vast/qc58/pub-db/MMWHS/processed", "MMWHS_training.h5", "MC-Net_toy","train")
create_dataset("/research/cbim/vast/qc58/pub-db/MMWHS/processed", "MMWHS_training.h5", "MC-Net_toy","test",1,5,10,False)
create_dataset("/research/cbim/vast/qc58/pub-db/MMWHS/processed", "MMWHS_training.h5", "MC-Net_toy","test",1,5,10,True)
create_dataset("/research/cbim/vast/qc58/pub-db/MMWHS/processed", "MMWHS_training.h5", "MC-Net_toy","test",1,20,10,True)
create_dataset("/research/cbim/vast/qc58/pub-db/MMWHS/processed", "MMWHS_training.h5", "MC-Net_toy","validate")



