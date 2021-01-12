import os
import numpy as np
import h5py
import cv2

def merge_h5(file_list, prefix, target):
    """
    Merge all hdf5 files in file_list to target file with prefix name id.
    :param file_list:
    :param prefix:
    :param target:
    :return:
    """

    h5_target = h5py.File(target, 'w')
    for idx, file_path in enumerate(file_list):
        source = h5py.File(file_path,'r')
        source_list = list(source['train'])
        source_prefix = prefix[idx]
        for data in source_list:
            print(f"save:{source_prefix}-{data} data")
            img = source[f'train/{data}/data'][()]
            label = source[f'train/{data}/label'][()]
            h5_target.create_dataset(f"train/{source_prefix}-{data}/data",data=img)
            h5_target.create_dataset(f"train/{source_prefix}-{data}/label",data=label)

        source.close()
    h5_target.close()


merge_h5(["/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2/General_format_BraTS18_train_three_center_0_2d_3ch.h5","/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/test_140/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch140.h5"],["real","syn"],"/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/realds0+synall.h5")

merge_h5(["/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2/General_format_BraTS18_train_three_center_1_2d_3ch.h5","/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/test_140/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch140.h5"],["","syn"],"/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/realds1+synall.h5")
#
merge_h5(["/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2/General_format_BraTS18_train_three_center_2_2d_3ch.h5","/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/test_140/brats_multilabel_perceptionloss100_hgglgg_resnet_9blocks_epoch140.h5"],["","syn"],"/freespace/local/qc58/dataset/BraTS2018/tmp/brats_AsynDGANv2_3db_exp11/Brats3db_resnet_3ch_9d/realds2+synall.h5")
