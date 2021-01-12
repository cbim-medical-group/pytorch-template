import os
import random

import numpy as np
import h5py


def read_slice_and_save(root, source_name, target_name, save_type="multi_modality"):

    type="train"
    source = h5py.File(os.path.join(root, source_name),'r')
    target = h5py.File(os.path.join(root, target_name), 'w')

    ids = source[type].keys()
    for id in ids:
        seg = source[f'{type}/{id}/label'][()]
        t1 = source[f'{type}/{id}/data/t1'][()]
        t2 = source[f'{type}/{id}/data/t2'][()]
        flair = source[f'{type}/{id}/data/flair'][()]
        t1ce = source[f'{type}/{id}/data/t1ce'][()]
        length = seg.shape[2]
        for i in range(length):
            seg_slice = seg[:,:,i]
            t1_slice = t1[:,:,i]
            t2_slice = t2[:,:,i]
            flair_slice = flair[:,:,i]
            t1ce_slice = t1ce[:,:,i]
            if seg_slice.sum()>0:
                print(f"Save slice: train/{id}-{i}/")
                target.create_dataset(f"{type}/{id}-{i}/data/t1", data=t1_slice)
                target.create_dataset(f"{type}/{id}-{i}/data/t2", data=t2_slice)
                target.create_dataset(f"{type}/{id}-{i}/data/flair", data=flair_slice)
                target.create_dataset(f"{type}/{id}-{i}/data/t1ce", data=t1ce_slice)
                target.create_dataset(f"{type}/{id}-{i}/label", data=seg_slice)



    source.close()
    target.close()


    # """
    #
    # :param root:
    # :param source_name:
    # :param target_name:
    # :param save_type: "multi_modality" or "multi_channel"
    # :return:
    # """
    # source = h5py.File(os.path.join(root, source_name),'r')
    # # target = h5py.File(os.path.join(root,target_name), 'w')
    # ids = list(source['train'])
    #
    # # random.shuffle(ids)
    # id_tumor_size_map = []
    # for i in ids:
    #     seg = source[f'train/{i}/label'][()]
    #     volume = seg[seg>0].sum()
    #     id_tumor_size_map.append((i, volume))
    #
    # id_tumor_size_map.sort(key=lambda x: x[1])
    # ids = [i[0] for i in id_tumor_size_map]
    #
    # target_files = []
    # for i in range(10):
    #     target = h5py.File(os.path.join(root,target_name+f"_{i}.h5"), 'w')
    #     target_files.append(target)
    #
    #
    # for indexes in range(len(ids)):
    #     idx = indexes//17
    #     cur_target = target_files[idx]
    #     id = ids[indexes]
    #     seg = source[f'train/{id}/label'][()]
    #     t1 = source[f'train/{id}/data/t1'][()]
    #     t2 = source[f'train/{id}/data/t2'][()]
    #     flair = source[f'train/{id}/data/flair'][()]
    #     t1ce = source[f'train/{id}/data/t1ce'][()]
    #     length = seg.shape[2]
    #     for i in range(length):
    #         seg_slice = seg[:,:,i]
    #         t1_slice = t1[:,:,i]
    #         t2_slice = t2[:,:,i]
    #         flair_slice = flair[:,:,i]
    #         t1ce_slice = t1ce[:,:,i]
    #         if seg_slice.sum()>0:
    #             print(f"Save slice: _{idx}.h5: train/{id}-{i}/")
    #             cur_target.create_dataset(f"train/{id}-{i}/data/t1", data=t1_slice)
    #             cur_target.create_dataset(f"train/{id}-{i}/data/t2", data=t2_slice)
    #             cur_target.create_dataset(f"train/{id}-{i}/data/flair", data=flair_slice)
    #             cur_target.create_dataset(f"train/{id}-{i}/data/t1ce", data=t1ce_slice)
    #             cur_target.create_dataset(f"train/{id}-{i}/label", data=seg_slice)
    #
    #
    # for i in range(10):
    #     target_files[i].close()

read_slice_and_save("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
                    "BraTS18_train.h5",
                    "BraTS18_train_2d.h5")


