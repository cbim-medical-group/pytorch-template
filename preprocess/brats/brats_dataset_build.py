"""
Build hdf5 files based on MURA datasets.

"""

import os
import sys
import random

import h5py
import nibabel as nib


def readlist(dcm_path, save_path, save_name):
    # Other:20, CBICA:88, TCIA: 102 Total: 210.
    # train:14+61+71 val:2+9+11 test:4+18+20  (7:1:2)
    dcm_folders = sorted(os.listdir(dcm_path))


    other_list = [name for name in dcm_folders if name.find("2013")>0]
    CBICA_list = [name for name in dcm_folders if name.find("CBICA")>0]
    TCIA_list = [name for name in dcm_folders if name.find("TCIA")>0]

    random.shuffle(other_list)
    random.shuffle(CBICA_list)
    random.shuffle(TCIA_list)
    assert len(other_list) == 20
    assert len(CBICA_list) == 88
    assert len(TCIA_list) == 102

    train_list = other_list[:14] + CBICA_list[:61] + TCIA_list[:71]
    val_list = other_list[14:16] + CBICA_list[61:70] + TCIA_list[71:82]
    test_list = other_list[16:] + CBICA_list[70:] + TCIA_list[82:]
    assert len(train_list) == 146
    assert len(val_list) == 22
    assert len(test_list) == 42

    type = ["train","val","test"]
    result_files = [h5py.File(os.path.join(save_path, save_name+"_"+type[0]+".h5"), "w"),
                    h5py.File(os.path.join(save_path, save_name+"_"+type[1]+".h5"), "w"),
                    h5py.File(os.path.join(save_path, save_name+"_"+type[2]+".h5"), "w"),]
    for idx, list in enumerate([train_list, val_list, test_list]):
        for data_id in list:
            if os.path.isdir(os.path.join(dcm_path, data_id)):
                sys.stdout.flush()
                flair = nib.load(os.path.join(dcm_path, data_id, f"{data_id}_flair.nii.gz")).get_fdata()
                seg = nib.load(os.path.join(dcm_path, data_id, f"{data_id}_seg.nii.gz")).get_fdata()
                t1 = nib.load(os.path.join(dcm_path, data_id, f"{data_id}_t1.nii.gz")).get_fdata()
                t1ce = nib.load(os.path.join(dcm_path, data_id, f"{data_id}_t1ce.nii.gz")).get_fdata()
                t2 = nib.load(os.path.join(dcm_path, data_id, f"{data_id}_t2.nii.gz")).get_fdata()

                result_files[idx].create_dataset(f"{type[idx]}/{data_id}/flair", data=flair, compression='gzip')
                result_files[idx].create_dataset(f"{type[idx]}/{data_id}/seg", data=seg, compression='gzip')
                result_files[idx].create_dataset(f"{type[idx]}/{data_id}/t1", data=t1, compression='gzip')
                result_files[idx].create_dataset(f"{type[idx]}/{data_id}/t1ce", data=t1ce, compression='gzip')
                result_files[idx].create_dataset(f"{type[idx]}/{data_id}/t2", data=t2, compression='gzip')

                print(f"***Finish create one database:{type[idx]}/{data_id},***")



    result_files[0].close()
    result_files[1].close()
    result_files[2].close()

readlist("/share_hd1/db/BRATS/2018/HGG","/share_hd1/db/BRATS/AsynDGANv2","BraTS18")

# readlist("/share_hd1/db/BRATS/2018/LGG","/share_hd1/db/BRATS/2018","BraTS18_LGG.h5",15)
