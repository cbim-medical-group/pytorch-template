import os
import numpy as np
import h5py
import cv2

def convert(h5_path, folder_path, type="train"):
    """
    Save the hdf5 file in h5_path to folder_path jpg files.
    :param h5_path:
    :param folder_path:
    :param type:
    :return:
    """
    f = h5py.File(h5_path,'r')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in f[f'{type}'].keys():
        img = f[f'{type}/{i}/data'][()]
        img = img[:1]
        img = np.moveaxis(img,0,-1)
        print(f"write image to:{i}.jpg")
        cv2.imwrite(os.path.join(folder_path,f"{i}.jpg"), img)

# convert("/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/test_140/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch140.h5",
#         "/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/test_140/images")

convert("/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_1db_exp9_fixbug/Brats3db_resnet_1ch/test_200/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch200.h5",
        "/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_1db_exp9_fixbug/Brats3db_resnet_1ch/test_200/images")

convert("/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_random_split_1ch_exp13_fixbug/Brats3db_resnet_random_split_1ch/test_200/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch200.h5",
        "/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_random_split_1ch_exp13_fixbug/Brats3db_resnet_random_split_1ch/test_200/images")

