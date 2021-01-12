import os

import h5py


def split_dataset(rootpath, source, target_pattern, split_number):
    source_h5 = h5py.File(os.path.join(rootpath, source),'r')



    for key in source_h5.keys():
        source_sub = source_h5[f'{key}']
        sub_list = list(source_sub.keys())
        sub_list_size = len(sub_list)//split_number
        sub_list_idx_arr = [sub_list[i:i + sub_list_size] for i in range(0, len(sub_list), sub_list_size)]

        for sub_list_idx in sub_list_idx_arr:
            sub_sub_list = sub_list[sub_list_idx]
            print(sub_sub_list, len(sub_sub_list))

split_dataset("/research/cbim/vast/qc58/pub-db/FashionMNIST","train_MNIST_fashionMNIST.h5","train_MNIST_fashionMNIST_#i#.h5",50)

