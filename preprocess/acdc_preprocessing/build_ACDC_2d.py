import os

import h5py


def extract_2d_slices_and_save(root, source, target):
    db = h5py.File(os.path.join(root, source),'r')
    target_db = h5py.File(os.path.join(root, target), 'w')
    for key in db['train'].keys():
        list = db[f'train/{key}'].keys()
        for id in list:
            if id != "data_without_label":
                volume = db[f"train/{key}/{id}/data"][()]
                label = db[f"train/{key}/{id}/label"][()]
                print(f"save data for train/{key}/{id}")
                for slice_idx in range(label.shape[2]):
                    if label[:,:,slice_idx].sum() >0:
                        target_db.create_dataset(f"train/{key}-{id}-{slice_idx}/data", data=volume[:, :, slice_idx])
                        target_db.create_dataset(f"train/{key}-{id}-{slice_idx}/label", data=label[:, :, slice_idx])
    target_db.close()
    db.close()

extract_2d_slices_and_save("/freespace/local/qc58/dataset/ACDC/processed","ACDC_train.h5","general_format_ACDC_train_2d.h5")
