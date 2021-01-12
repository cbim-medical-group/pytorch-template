import json
import os
import re

import h5py


def save_sax_lax(sources, target_path, sax_target, lax_target):
    target_sax = h5py.File(os.path.join(target_path, sax_target), 'w')
    target_lax = h5py.File(os.path.join(target_path, lax_target), 'w')
    for source in sources:
        source_f = h5py.File(source,'r')
        all_files = {}
        for key in list(source_f['train']):
            desc = source_f[f'train/{key}/data'].attrs['description']

            print(f"process entry:{key}, desc:{desc}")
            re_key = re.search(f'study:(.*)-series:(.*)', key)
            study_id = re_key.group(1)
            series_id = re_key.group(2)

            verify_sax = re.compile(r'(lv sax fiesta)|(short axis)', re.IGNORECASE)
            verify_lax = re.compile(f'(lv lax fiesta)|(long axis)', re.IGNORECASE)
            if verify_sax.search(desc):
                target_sax.create_dataset(f"train/{key}/data", data=source_f[f'train/{key}/data'][()], compression="gzip")
                for attr in list(source_f[f'train/{key}/data'].attrs):
                    target_sax[f"train/{key}/data"].attrs[attr] =source_f[f'train/{key}/data'].attrs[attr]
            if verify_lax.search(desc):
                target_lax.create_dataset(f"train/{key}/data", data=source_f[f'train/{key}/data'][()], compression="gzip")
                for attr in list(source_f[f'train/{key}/data'].attrs):
                    target_lax[f"train/{key}/data"].attrs[attr] =source_f[f'train/{key}/data'].attrs[attr]

    target_sax.close()
    target_lax.close()

save_sax_lax(["/dresden/users/qc58/work/ATMI/output/1_tmp/Export-1-Derivate_Cardial_MRI_DB-1593059313-0.h5","/dresden/users/qc58/work/ATMI/output/1_tmp/Export-1-Derivate_Cardial_MRI_DB-1593099659-1.h5"],"/dresden/users/qc58/work/ATMI/output/1_tmp/","Export-1-Derivate_SAX.h5","Export-1-Derivate_LAX.h5")