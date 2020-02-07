
import os, json, glob
from skimage import io
from tqdm import tqdm
import numpy as np
import random

import h5py


def build_h5(data_dir, save_dir, save_name):
    with open('{:s}/train_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, test_list = data_list['train'], data_list['testA']

    result_file = h5py.File(os.path.join(save_dir, save_name), "w")

    # train imgs are 250 x 250 resolution extracted from original images
    print('Processing training files')
    for img_name in tqdm(train_list):
        name = img_name.split('.')[0]
        for i in range(4):   # 16 patches for each large image
            # images
            img = io.imread('{:s}/patches_286/images/{:s}_{:d}.png'.format(data_dir, name, i))
            result_file.create_dataset('train/images/{:s}_{:d}'.format(name, i), data=img)
            # labels
            label = io.imread('{:s}/patches_286/labels/{:s}_{:d}.png'.format(data_dir, name, i))
            result_file.create_dataset('train/labels/{:s}_{:d}'.format(name, i), data=label)
            # ternary labels (for segmentation)
            label_ternary = io.imread('{:s}/patches_286/labels_ternary/{:s}_{:d}_label.png'.format(data_dir, name, i))
            result_file.create_dataset('train/labels_ternary/{:s}_{:d}'.format(name, i), data=label_ternary)
            # weight maps (for segmentation)
            weight_map = io.imread('{:s}/patches_286/weight_maps/{:s}_{:d}_weight.png'.format(data_dir, name, i))
            result_file.create_dataset('train/weight_maps/{:s}_{:d}'.format(name, i), data=weight_map)

    # test imgs are original 1000 x 1000 resolution
    print('Processing test files')
    for img_name in tqdm(test_list):
        name = img_name.split('.')[0]
        # images
        img = io.imread('{:s}/images/{:s}.png'.format(data_dir, name))
        result_file.create_dataset('test/images/{:s}'.format(name), data=img)
        # labels
        label = io.imread('{:s}/labels/{:s}.png'.format(data_dir, name))
        result_file.create_dataset('test/labels/{:s}'.format(name), data=label)
        # instance labels (for segmentation)
        label_instance = io.imread('{:s}/labels_instance/{:s}.png'.format(data_dir, name))
        result_file.create_dataset('test/labels_instance/{:s}'.format(name), data=label_instance)

    result_file.close()


def build_h5_for_segmentation(data_dir, save_dir):
    with open('{:s}/train_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, test_list, test2_list = data_list['train'], data_list['testA'], data_list['testB']

    train_file_all = h5py.File('{:s}/train_all.h5'.format(save_dir), "w")
    train_file_breast = h5py.File('{:s}/train_breast.h5'.format(save_dir), "w")
    train_file_kidney = h5py.File('{:s}/train_kidney.h5'.format(save_dir), "w")
    train_file_liver = h5py.File('{:s}/train_liver.h5'.format(save_dir), "w")
    train_file_prostate = h5py.File('{:s}/train_prostate.h5'.format(save_dir), "w")
    test_file = h5py.File('{:s}/test.h5'.format(save_dir), "w")
    # test_file = h5py.File('{:s}/test2.h5'.format(save_dir), "w")

    # train imgs are 250 x 250 resolution extracted from original images
    print('Processing all training files')
    _dump_training_files(train_file_all, data_dir, train_list)

    breast_file_list = [filename for filename in train_list if 'Breast' in filename]
    kidney_file_list = [filename for filename in train_list if 'Kidney' in filename]
    liver_file_list = [filename for filename in train_list if 'Liver' in filename]
    prostate_file_list = [filename for filename in train_list if 'Prostate' in filename]
    print('Processing subset training files')
    _dump_training_files(train_file_breast, data_dir, breast_file_list)
    _dump_training_files(train_file_kidney, data_dir, kidney_file_list)
    _dump_training_files(train_file_liver, data_dir, liver_file_list)
    _dump_training_files(train_file_prostate, data_dir, prostate_file_list)

    # test imgs are original 1000 x 1000 resolution
    print('Processing test files')
    for img_name in tqdm(test_list):
        name = img_name.split('.')[0]
        # images
        img = io.imread('{:s}/original_images/{:s}.png'.format(data_dir, name))
        test_file.create_dataset('images/{:s}'.format(name), data=img)
        # labels
        label = io.imread('{:s}/labels/{:s}.png'.format(data_dir, name))
        test_file.create_dataset('labels/{:s}'.format(name), data=label)
        # ternary labels (for segmentation)
        label_ternary = io.imread('{:s}/labels_ternary/{:s}_label.png'.format(data_dir, name))
        test_file.create_dataset('labels_ternary/{:s}'.format(name), data=label_ternary)
        # weight maps (for segmentation)
        weight_map = io.imread('{:s}/weight_maps/{:s}_weight.png'.format(data_dir, name))
        test_file.create_dataset('weight_maps/{:s}'.format(name), data=weight_map)
        # instance labels (for segmentation)
        label_instance = io.imread('{:s}/labels_instance/{:s}.png'.format(data_dir, name))
        test_file.create_dataset('labels_instance/{:s}'.format(name), data=label_instance)

    train_file_all.close()
    train_file_breast.close()
    train_file_kidney.close()
    train_file_liver.close()
    train_file_prostate.close()
    test_file.close()


def _dump_training_files(h5_file, data_dir, img_names):
    for img_name in tqdm(img_names):
        name = img_name.split('.')[0]
        for i in range(16):   # 16 patches for each large image
            # images
            img = io.imread('{:s}/patches_256/original_images/{:s}_{:d}.png'.format(data_dir, name, i))
            h5_file.create_dataset('images/{:s}_{:d}'.format(name, i), data=img)
            # labels
            label = io.imread('{:s}/patches_256/labels/{:s}_{:d}.png'.format(data_dir, name, i))
            h5_file.create_dataset('labels/{:s}_{:d}'.format(name, i), data=label)
            # ternary labels (for segmentation)
            label_ternary = io.imread('{:s}/patches_256/labels_ternary/{:s}_{:d}_label.png'.format(data_dir, name, i))
            h5_file.create_dataset('labels_ternary/{:s}_{:d}'.format(name, i), data=label_ternary)
            # weight maps (for segmentation)
            weight_map = io.imread('{:s}/patches_256/weight_maps/{:s}_{:d}_weight.png'.format(data_dir, name, i))
            h5_file.create_dataset('weight_maps/{:s}_{:d}'.format(name, i), data=weight_map)


def split_patches(data_dir, save_dir, post_fix=None):
    import math
    """ split large image into small patches """
    os.makedirs(save_dir, exist_ok=True)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 256
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def extract_imgs(h5_file_path, save_path):
    h5_file = h5py.File(h5_file_path, 'r')
    for key in h5_file.keys():
        os.makedirs('{:s}/{:s}'.format(save_path, key), exist_ok=True)
        for file_key in h5_file[key].keys():
            img = h5_file['{:s}/{:s}'.format(key, file_key)][()]
            # if key == 'images':
            #     img = (img + 1) / 2.0 * 255
            #     img = img.astype(np.uint8)
            # if len(img.shape) > 2:
            #     img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
            io.imsave('{:s}/{:s}/{:s}.png'.format(save_path, key, file_key), img)


def prepare_seg_data_from_syn_h5(h5_file_path, save_path):
    file_name = h5_file_path.split('/')[-1]
    result_file = h5py.File('{:s}/{:s}.h5'.format(save_path, file_name.split('.')[0]), 'w')
    h5_file = h5py.File(h5_file_path, 'r')
    # images
    for file_key in h5_file['images'].keys():
        # if file_key.split('.')[0][-1] == '2':  # skip the same file ending with _2
        #     continue
        if file_key.split('.')[0][-1] != '0':  # only select file ending with _0
            continue
        img = h5_file['images/{:s}'.format(file_key)][()]
        img = (img + 1) / 2.0 * 255
        img = img.astype(np.uint8)
        result_file.create_dataset('images/{:s}'.format(file_key), data=img)

    # labels_ternary
    for file_key in h5_file['label_ternary'].keys():
        # if file_key.split('.')[0][-1] == '2':  # skip the same file ending with _2
        #     continue
        if file_key.split('.')[0][-1] != '0':  # only select file ending with _0
            continue
        img = h5_file['label_ternary/{:s}'.format(file_key)][()]
        result_file.create_dataset('labels_ternary/{:s}'.format(file_key), data=img)

    # weight_maps
    for file_key in h5_file['weight_map'].keys():
        # if file_key.split('.')[0][-1] == '2':  # skip the same file ending with _2
        #     continue
        if file_key.split('.')[0][-1] != '0':  # only select file ending with _0
            continue
        img = h5_file['weight_map/{:s}'.format(file_key)][()]
        result_file.create_dataset('weight_maps/{:s}'.format(file_key), data=img)

    result_file.close()
    h5_file.close()

# split_patches('/share_hd1/db/Nuclei/original/original_images', '/share_hd1/db/Nuclei/original/patches_256/original_images')
# split_patches('/share_hd1/db/Nuclei/original/labels', '/share_hd1/db/Nuclei/original/patches_286/labels')
# split_patches('/share_hd1/db/Nuclei/original/labels_ternary', '/share_hd1/db/Nuclei/original/patches_286/labels_ternary', post_fix='label')
# split_patches('/share_hd1/db/Nuclei/original/weight_maps', '/share_hd1/db/Nuclei/original/patches_286/weight_maps', post_fix='weight')

# build_h5('/share_hd1/db/Nuclei/original', '/share_hd1/db/Nuclei', 'Nuclei_all_286_new.h5')
build_h5_for_segmentation('/share_hd1/db/Nuclei/original', '/share_hd1/db/Nuclei/for_seg_256_new')
# extract_imgs('/share_hd1/db/Nuclei/for_seg_256_new/train_all.h5', '/share_hd1/db/Nuclei/for_seg_256_new/imgs_all')

# extract_imgs('/share_hd1/db/Nuclei/syn_whole/Nuclei_Dadgan_resnet_9blocks_epoch300_512.h5', '/share_hd1/db/Nuclei/syn_whole/imgs_dadgan_512')

# prepare_seg_data_from_syn_h5('/share_hd1/db/Nuclei/syn_whole/Nuclei_Dadgan_resnet_9blocks_epoch300_withoutL1.h5', '/share_hd1/db/Nuclei/for_seg_256')
# extract_imgs('/share_hd1/db/Nuclei/for_seg_256/Nuclei_Dadgan_resnet_9blocks_epoch300_withoutL1.h5', '/share_hd1/db/Nuclei/for_seg_256/imgs_without_L1_epoch300')


# prepare_seg_data_from_syn_h5('/share_hd1/db/Nuclei/syn_whole/Nuclei_all_resnet_9blocks_epoch400_x5.h5', '/share_hd1/db/Nuclei/for_seg_256')