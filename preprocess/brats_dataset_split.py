"""
Build hdf5 files based on BraTS datasets.

"""

import os
import sys
import random
import numpy as np
from tqdm import tqdm
import h5py
import scipy.misc as misc


def partition(orig_file_path, save_path, save_prefix, num_partition, method_partition):
    os.makedirs(save_path, exist_ok=True)
    orig_file = h5py.File(orig_file_path, "r")
    result_files = [h5py.File('{:s}/{:s}_{:s}_{:d}.h5'.format(save_path, save_prefix, method_partition, i), "w") for i in range(num_partition)]

    train_set = orig_file['train']
    keys = list(train_set.keys())

    # same test set for all subsets
    for i in range(num_partition):
        result_files[i].create_group('/train')
        # orig_file.copy('/test', result_files[i])

    num_train_cases = len(keys)
    chunk_size = num_train_cases // num_partition
    if method_partition == 'random':
        random.seed(1)
        random.shuffle(keys)
        counter = 0
        for key in keys:
            orig_file.copy('/train/{:s}'.format(key), result_files[counter//chunk_size]['/train'])
            counter += 1
    if method_partition == 'tumor_size':
        size_dict = compute_tumor_size(train_set)
        print(size_dict)
        counter = 0
        for key, _ in size_dict:
            orig_file.copy('/train/{:s}'.format(key), result_files[counter // chunk_size]['/train'])
            counter += 1

    # close datasets
    orig_file.close()
    for i in range(num_partition):
        result_files[i].close()


def compute_tumor_size(train_set):
    keys = list(train_set.keys())
    size_dict = {}
    for key in keys:
        label = train_set['{:s}/seg'.format(key)][()]
        tumor_size = np.count_nonzero(label)
        size_dict[key] = tumor_size
    size_dict = sorted(size_dict.items(), key=lambda kv: kv[1])
    return size_dict


def save_png_for_seg(data_path, save_path):
    data = h5py.File(data_path, 'r')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    train_data, test_data = data['train'], data['test']
    _save_png_with_tumor(train_data, '{:s}/train'.format(save_path))
    _save_png_with_tumor(test_data, '{:s}/test'.format(save_path))


def _save_png_with_tumor(dataset, save_path):
    os.makedirs('{:s}/images'.format(save_path), exist_ok=True)
    os.makedirs('{:s}/labels'.format(save_path), exist_ok=True)
    keys = dataset.keys()
    for key in keys:
        dcm = dataset[f"{key}/t2"][()]
        label = dataset[f"{key}/seg"][()]

        for i in range(label.shape[2]):
            slice_label = label[:, :, i]
            if np.count_nonzero(slice_label) < 10:
                continue

            slice_dcm = dcm[:, :, i]
            print('{:f}\t{:f}'.format(slice_dcm.min(), slice_dcm.max()))
            slice_dcm = slice_dcm * ((pow(2, 8) - 1) / slice_dcm.max())
            slice_dcm = slice_dcm.astype("uint8")
            slice_label = slice_label.astype("uint8")
            slice_label[slice_label > 0] = 255
            misc.imsave('{:s}/images/{:s}_{:d}.png'.format(save_path, key, i), slice_dcm)
            misc.imsave('{:s}/labels/{:s}_{:d}.png'.format(save_path, key, i), slice_label)
        print(f"Saved images and labels of {key} to: {save_path}")


def save_h5_file_for_seg(data_path, save_path):
    data = h5py.File(data_path, 'r')
    os.makedirs(save_path, exist_ok=True)
    train_data = data['train']
    _save_h5_with_tumor(train_data, '{:s}/train.h5'.format(save_path))
    if 'test' in data.keys():
        test_data = data['test']
        _save_h5_with_tumor(test_data, '{:s}/test.h5'.format(save_path))
    data.close()


def _save_h5_with_tumor(dataset, save_filepath):
    h5_file = h5py.File(save_filepath, 'w')
    h5_file.create_group('images')
    h5_file.create_group('labels')
    keys = dataset.keys()
    print(list(keys))
    for key in tqdm(keys):
        dcm = dataset[f"{key}/t2"][()]
        label = dataset[f"{key}/seg"][()]

        for i in range(label.shape[2]):
            slice_label = label[:, :, i]
            if np.count_nonzero(slice_label) < 10:
                continue

            slice_dcm = dcm[:, :, i]
            # print('{:f}\t{:f}'.format(slice_dcm.min(), slice_dcm.max()))
            slice_dcm = slice_dcm * ((pow(2, 8) - 1) / slice_dcm.max())
            slice_dcm = slice_dcm.astype("uint8")
            slice_dcm = slice_dcm[:, :, None].repeat(3, axis=2)
            slice_label = slice_label.astype("uint8")
            slice_label[slice_label > 0] = 255

            h5_file.create_dataset('/images/{:s}_{:d}'.format(key, i), data=slice_dcm)
            h5_file.create_dataset('/labels/{:s}_{:d}'.format(key, i), data=slice_label)

        # print(f"Saved images and labels of {key} to: {save_filepath}")
    h5_file.close()


def combine_data_set(data_path1, data_path2, save_path):
    data1 = h5py.File(data_path1, 'r')
    data2 = h5py.File(data_path2, 'r')
    h5_file = h5py.File(save_path, 'w')
    h5_file.create_group('train')
    h5_file.create_group('test')
    for key in tqdm(data1['train'].keys()):
        h5_file.create_group('train/HGG_{:s}'.format(key))
        for subkey in data1['train/{:s}'.format(key)].keys():
            h5_file.create_dataset('train/HGG_{:s}/{:s}'.format(key, subkey),
                                   data=data1['train/{:s}/{:s}'.format(key, subkey)][()])
    for key in tqdm(data2['train'].keys()):
        h5_file.create_group('train/LGG_{:s}'.format(key))
        for subkey in data2['train/{:s}'.format(key)].keys():
            h5_file.create_dataset('train/LGG_{:s}/{:s}'.format(key, subkey),
                                   data=data2['train/{:s}/{:s}'.format(key, subkey)][()])

    for key in tqdm(data1['test'].keys()):
        h5_file.create_group('test/HGG_{:s}'.format(key))
        for subkey in data1['test/{:s}'.format(key)].keys():
            h5_file.create_dataset('test/HGG_{:s}/{:s}'.format(key, subkey),
                                   data=data1['test/{:s}/{:s}'.format(key, subkey)][()])
    for key in tqdm(data2['test'].keys()):
        h5_file.create_group('test/LGG_{:s}'.format(key))
        for subkey in data2['test/{:s}'.format(key)].keys():
            h5_file.create_dataset('test/LGG_{:s}/{:s}'.format(key, subkey),
                                   data=data2['test/{:s}/{:s}'.format(key, subkey)][()])
    h5_file.close()
    data1.close()
    data2.close()


def test():
    h5_file1 = h5py.File('/share_hd1/db/BRATS/brats18_for_seg/BraTS18_all_v1/train.h5', 'r')
    h5_file2 = h5py.File('/share_hd1/db/BRATS/brats18_for_seg/BraTS18_all/train.h5', 'r')

    len1 = len(list(h5_file1['images']))
    len2 = len(list(h5_file2['images']))
    print('{:d}\t{:d}'.format(len1, len2))


# test()


# partition("/share_hd1/db/BRATS/2018/BraTS18.h5", "/share_hd1/db/BRATS/2018/tumor_size_split_10", "BraTS18", num_partition=10,
#           method_partition='tumor_size')
# save_png_for_seg("/share_hd1/db/BRATS/2018/BraTS18_random_1.h5", "/share_hd1/db/BRATS/2018/BraTS18_random_1")

# save_h5_file_for_seg("/share_hd1/db/BRATS/2018/BraTS18.h5", "/share_hd1/db/BRATS/brats18_for_seg/BraTS18_all_v2")

# for i in range(10):
#     save_h5_file_for_seg("/share_hd1/db/BRATS/2018/tumor_size_split_10/BraTS18_tumor_size_{:d}.h5".format(i),
#                          "/share_hd1/db/BRATS/brats18_for_seg/tumor_size_split_10/BraTS18_tumor_size_{:d}".format(i))


# --- for LGG data --- #
# save_h5_file_for_seg("/share_hd1/db/BRATS/2018/BraTS18_LGG.h5", "/share_hd1/db/BRATS/brats18_for_seg/LGG/BraTS18_LGG_all")

# combine_data_set("/share_hd1/db/BRATS/2018/BraTS18.h5", "/share_hd1/db/BRATS/2018/BraTS18_LGG.h5",
#                   "/share_hd1/db/BRATS/2018/BraTS18_HGG_LGG.h5")
save_h5_file_for_seg("/share_hd1/db/BRATS/2018/BraTS18_HGG_LGG.h5", "/share_hd1/db/BRATS/brats18_for_seg/BraTS18_all_HGG_LGG")


