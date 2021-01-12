import os

import h5py
from skimage.transform import resize


def resize_h5(path, source_file, target_file, target_size, type="train"):
    """
    Resize the entry of source h5 to specific size.
    :param source:
    :param target:
    :return:
    """
    source_h5 = h5py.File(os.path.join(path, source_file), 'r')
    target_h5 = h5py.File(os.path.join(path, target_file), 'w')

    for entry in list(source_h5[type]):
        print(f"entry:{entry}")
        # data = source_h5[f"{type}/{entry}/data"][()]
        label = source_h5[f"{type}/{entry}/label"][()]
        # new_data = resize(data, target_size, order=1, preserve_range=True)
        new_label = resize(label, target_size, order=0, preserve_range=True)
        # target_h5.create_dataset(f"{type}/{entry}/data", data=new_data, compression="gzip")
        target_h5.create_dataset(f"{type}/{entry}/label", data=new_label, compression="gzip")

    source_h5.close()
    target_h5.close()

resize_h5("/ajax/users/qc58/work/projects/pytorch-template/saved/test_results/ACDC_segmentation_exp1","test_result-ACDC_segmentation_exp1.h5","test_result-ACDC_segmentation_exp1_512x512.h5", (512,512))


