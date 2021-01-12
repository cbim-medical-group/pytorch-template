import random

import h5py
import pydicom
import os
from cv2 import cv2
import numpy as np
from utils import getListOfFiles


def create_h5_dataset(root_path, type="la"):
    la_lists = list_all_data(os.path.join(root_path, "raw"))
    random.shuffle(la_lists)
    test_num = round(len(la_lists)*0.2)
    test_la_lists = la_lists[:test_num]
    train_la_lists = la_lists[test_num:]

    h5_train = h5py.File(f"{root_path}/process/cap_train.h5",'w')
    h5_test = h5py.File(f"{root_path}/process/cap_test.h5",'w')
    all_mean = []
    all_std = []
    for idx, case in enumerate(train_la_lists):
        print(f"process train list({idx}/{len(train_la_lists)})")
        dcm_path = case[2]
        data = h5_train.create_dataset(f"train/{idx}/data", data=case[0], compression='gzip')
        data.attrs['dcm_path'] = dcm_path
        h5_train.create_dataset(f"train/{idx}/label", data=case[1], compression='gzip')
        all_mean.append(case[4])
        all_std.append(case[5])

    for idx, case in enumerate(test_la_lists):
        print(f"process test list({idx}/{len(test_la_lists)})")
        dcm_path = case[2]
        id = dcm_path[-23:-4]
        h5_test.create_dataset(f"val/{id}/data", data=case[0], compression='gzip')
        h5_test.create_dataset(f"val/{id}/label", data=case[1], compression='gzip')


    h5_train.create_dataset(f"mean", data=np.average(all_mean))
    h5_train.create_dataset(f"std", data=np.average(all_std))
    h5_train.close()
    h5_test.close()
    print(f"mean:{np.average(all_mean)}, std:{np.average(all_std)}")


def list_all_data(root_path):
    all_data_list = []
    dcm_data_path = getListOfFiles(root_path, [".dcm"])
    dcm_data_path = [path for path in dcm_data_path if "la" in path.lower()]
    mask_data_path = [path.replace('dcm','png') for path in dcm_data_path]
    for idx, dcm_path in enumerate(dcm_data_path):
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array
        # arr = (arr/arr.max())*255
        mask_path = mask_data_path[idx]
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        mask[mask>0]=255
        # cv2.imwrite(f"./imgs/{mask_path[-23:]}_mask.png", mask)
        # cv2.imwrite(f"./imgs/{mask_path[-23:]}_dcm.png", arr)
        all_data_list.append((arr, mask, dcm_path, mask_path, arr.mean(), arr.std()))
        print(f"finish process:{mask_path}...({idx}/{len(dcm_data_path)})")
    return all_data_list

create_h5_dataset("/research/cbim/vast/qc58/pub-db/CAP")

