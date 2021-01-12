import os
import random
import re

import numpy as np
import h5py


def read_slice_and_save(root, source_name, target_name, save_type="train"):
    """

    :param root:
    :param source_name:
    :param target_name:
    :param save_type: "multi_modality" or "multi_channel"
    :return:
    """
    source = h5py.File(os.path.join(root, source_name+".h5"),'r')
    # target = h5py.File(os.path.join(root,target_name), 'w')
    ids = list(source[save_type])

    # Need data: 1. all 1ch_t2, train, val, test 2. all 3ch_t1t2flair, train, val, test. 3. all 3ch_t1ct2flair
    # 4. subset 1/10 1ch_t2, train
    # 5. AdynDGAN exp13_t2.
    # 6. real CBICA 3ch_t1t2flair, 7. real TCIA 3ch_t1t2flair 8. real Other 3ch_t1t2flair
    # 9. AsynDGAN exp11_t1t2flair
    # 10. Syn+Real-CBICA t1t2flair 11. Syn+real-TCIA t1t2flair 12. Syn+real-Other t1t2flair
    # 13. real CBICA_naT2, 14. real TCIA_naFlair 15. real Other_naT1c
    # 16. completed-CBICA_synT2, 17. completed-TCIA_synFlair, 18. complete-Other_synT1c


    target_all_t2 = h5py.File(os.path.join(root, target_name+f"_{save_type}"+"_all_t2.h5"), "w")
    target_all_t1t2flair = h5py.File(os.path.join(root, target_name+f"_{save_type}"+"_all_t1t2flair.h5"), "w")
    target_all_t1ct2flair = h5py.File(os.path.join(root, target_name + f"_{save_type}" + "_all_t1ct2flair.h5"), "w")

    for key in ids:
        seg = source[f"{save_type}/{key}/seg"][()]
        t1 = source[f"{save_type}/{key}/t1"][()]
        t1c = source[f"{save_type}/{key}/t1ce"][()]
        t2 = source[f"{save_type}/{key}/t2"][()]
        flair = source[f"{save_type}/{key}/flair"][()]

        length = seg.shape[2]
        for i in range(length):
            seg_slice = seg[:,:,i]
            t1_slice = t1[:,:,i]
            t2_slice = t2[:,:,i]
            flair_slice = flair[:,:,i]
            t1c_slice = t1c[:,:,i]
            if np.count_nonzero(seg_slice) >10:
            # if seg_slice.sum() > 10:
                print(f"Save slice({np.count_nonzero(seg_slice)}):  {save_type}/{key}-{i}/")
                # target_all_t2.create_dataset(f"train/{key}-{i}/data/t1", data=t1_slice, compression='gzip')
                target_all_t2.create_dataset(f"{save_type}/{key}-{i}/data", data=t2_slice, compression='gzip')
                # target_all_t2.create_dataset(f"train/{key}-{i}/data/flair", data=flair_slice, compression='gzip')
                # target_all_t2.create_dataset(f"train/{key}-{i}/data/t1ce", data=t1c_slice, compression='gzip')
                target_all_t2.create_dataset(f"{save_type}/{key}-{i}/label", data=seg_slice, compression='gzip')

                data = np.stack([t1_slice,t2_slice,flair_slice])
                target_all_t1t2flair.create_dataset(f"{save_type}/{key}-{i}/data", data=data, compression='gzip')
                target_all_t1t2flair.create_dataset(f"{save_type}/{key}-{i}/label", data=seg_slice, compression='gzip')

                data2 = np.stack([t1c_slice, t2_slice, flair_slice])
                target_all_t1ct2flair.create_dataset(f"{save_type}/{key}-{i}/data", data=data2, compression='gzip')
                target_all_t1ct2flair.create_dataset(f"{save_type}/{key}-{i}/label", data=seg_slice, compression='gzip')
            elif np.count_nonzero(seg_slice) >0:
                print(f"exclude({np.count_nonzero(seg_slice)}) {key}-{i}")

    target_all_t2.close()
    target_all_t1t2flair.close()
    target_all_t1ct2flair.close()
    source.close()

    # random.shuffle(ids)
    # id_tumor_size_map = []
    # for i in ids:
    #     seg = source[f'train/{i}/label'][()]
    #     volume = seg[seg>0].sum()
    #     id_tumor_size_map.append((i, volume))
    #
    # id_tumor_size_map.sort(key=lambda x: x[1])
    # ids = [i[0] for i in id_tumor_size_map]

    # target_files = []
    # # for i in range(3):
    # #     target = h5py.File(os.path.join(root,target_name+f"_{i}.h5"), 'w')
    # #     target_files.append(target)
    # target = h5py.File(os.path.join(root, target_name+".h5"), 'w')
    # target_files.append(target)

    # for indexes in range(len(ids)):
    #     idx = indexes//17
    #     cur_target = target_files[idx]
    #     key = ids[indexes]
    #     seg = source[f'train/{key}/label'][()]
    #     t1 = source[f'train/{key}/data/t1'][()]
    #     t2 = source[f'train/{key}/data/t2'][()]
    #     flair = source[f'train/{key}/data/flair'][()]
    #     t1ce = source[f'train/{key}/data/t1ce'][()]
    #     length = seg.shape[2]
    #     for i in range(length):
    #         seg_slice = seg[:,:,i]
    #         t1_slice = t1[:,:,i]
    #         t2_slice = t2[:,:,i]
    #         flair_slice = flair[:,:,i]
    #         t1ce_slice = t1ce[:,:,i]
    #         if seg_slice.sum()>0:
    #             print(f"Save slice: _{idx}.h5: train/{key}-{i}/")
    #             cur_target.create_dataset(f"train/{key}-{i}/data/t1", data=t1_slice)
    #             cur_target.create_dataset(f"train/{key}-{i}/data/t2", data=t2_slice)
    #             cur_target.create_dataset(f"train/{key}-{i}/data/flair", data=flair_slice)
    #             cur_target.create_dataset(f"train/{key}-{i}/data/t1ce", data=t1ce_slice)
    #             cur_target.create_dataset(f"train/{key}-{i}/label", data=seg_slice)
    #
    #
    # for i in range(10):
    #     target_files[i].close()



    # for key in ids:
    #
    #     target = target_files[0]
    #     print(f"{key} - Other")
    #
    #     # if key.find("CBICA") != -1:
    #     #     target = target_files[1]
    #     #     print(f"{key} - CBICA")
    #     # elif key.find("TCIA") != -1:
    #     #     target = target_files[2]
    #     #     print(f"{key} - TCIA")
    #     # else:
    #     #     target = target_files[0]
    #     #     print(f"{key} - Other")
    #
    #     seg = source[f'{save_type}/{key}/label'][()]
    #     t1 = source[f'{save_type}/{key}/data/t1'][()]
    #     t2 = source[f'{save_type}/{key}/data/t2'][()]
    #     flair = source[f'{save_type}/{key}/data/flair'][()]
    #     t1ce = source[f'{save_type}/{key}/data/t1ce'][()]
    #     length = seg.shape[2]
    #     for i in range(length):
    #         seg_slice = seg[:,:,i]
    #         t1_slice = t1[:,:,i]
    #         t2_slice = t2[:,:,i]
    #         flair_slice = flair[:,:,i]
    #         t1ce_slice = t1ce[:,:,i]
    #         if seg_slice.sum()>0:
    #             print(f"Save slice: train/{key}-{i}/")
    #             data = np.stack([t1ce_slice,t2_slice,flair_slice])
    #             target.create_dataset(f"{save_type}/{key}-{i}/data", data=data, compression='gzip')
    #             # target.create_dataset(f"train/{key}-{i}/data/t1", data=t1_slice)
    #             # target.create_dataset(f"train/{key}-{i}/data/t2", data=t2_slice)
    #             # target.create_dataset(f"train/{key}-{i}/data/flair", data=flair_slice)
    #             # target.create_dataset(f"train/{key}-{i}/data/t1ce", data=t1ce_slice)
    #             target.create_dataset(f"{save_type}/{key}-{i}/label", data=seg_slice)
    #
    #
    #
    # source.close()
    # for i in range(1):
    #     target_files[i].close()

# read_slice_and_save("/freespace/local/qc58/dataset/BraTS2018/AsynDGANv2",
#                     "BraTS18_train.h5",
#                     "General_format_BraTS18_train_three_center_2d_3ch_new")

read_slice_and_save("/share_hd1/db/BRATS/AsynDGANv2",
                    "BraTS18_train",
                    "General_format_BraTS18", "train")

