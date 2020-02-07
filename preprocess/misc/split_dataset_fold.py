import json

import h5py
import os


def find_group_id(split_ids_list, orig_file):
    for group_id in split_ids_list:
        for file_id_type in split_ids_list[group_id]:
            file_id = file_id_type.split("|")[0]
            patient_type = file_id_type.split("|")[1]
            if file_id in orig_file :
                return group_id, patient_type
    return None


def split_fold(split_ids_file, data_folder, split_source_file, target_file):
    """
    Split folds from train/***/[data|label] to 'group_id'/***/[data|label]
    :param split_ids_file: the split split_ids file should be like this format: {'group1':['id1','id2',...],'group2':['id3',...]}
    :param data_folder:
    :param split_source_file:
    :param target_file:
    :return:
    """
    with open(split_ids_file) as f:
        split_ids = json.load(f)

        orig_file = h5py.File(os.path.join(data_folder, split_source_file), 'r')

        if hasattr(orig_file, 'train'):
            orig_file = orig_file['train']

        target = h5py.File(os.path.join(data_folder, target_file),'w')
        for file_id in list(orig_file):
            group_id, patient_type = find_group_id(split_ids, file_id)
            print(f"group:{group_id}, patient type:{patient_type}, file_id:{file_id}")
            data = orig_file[f"{file_id}/data"][()]
            label = orig_file[f"{file_id}/label"][()]
            bbox = orig_file[f"{file_id}/bbox"][()]
            target.create_dataset(f"{group_id}/{file_id}/data", data=data)
            label = target.create_dataset(f"{group_id}/{file_id}/label", data=label)
            label.attrs["bbox"] = bbox
            label.attrs['patient_type'] = patient_type

        target.close()
        orig_file.close()


def convert_split_files(split_files_folder, split_files_ids, data_file_path, save_split_list_path,
                        save_split_file_name):
    result_files = {}
    for split_id in split_files_ids:
        id_list = []
        with open(os.path.join(split_files_folder, split_id + ".txt")) as f:
            for line in f:
                file_name = "patient" + ("00" + line)[-4:-1]

                with open(os.path.join(data_file_path, file_name, "Info.cfg")) as cfg:
                    line_num = 1
                    for line in cfg:
                        if line_num == 3:
                            file_name += "|" + line[7:-1]
                        line_num += 1
                id_list.append(file_name)

        result_files[split_id] = id_list

    with open(os.path.join(save_split_list_path, save_split_file_name), "w+") as wf:
        wf.write(json.dumps(result_files))
        # wf.write(result_files)

    # return result_files


# convert_split_files("/Users/qichang/PycharmProjects/pytorch-template/preprocess/misc/",
#                     ['train_0', 'train_1', 'train_2', 'train_3', 'train_4'],
#                     "/Users/qichang/Downloads/training/",
#                     "/Users/qichang/PycharmProjects/pytorch-template/preprocess/misc/", "ACDC_5fold_reference.json")

split_fold("/share_hd1/projects/code_seg_qi/preprocess/misc/ACDC_5fold_reference.json",
           "/share_hd1/projects/code_seg_qi/db/ACDC/processed", "ACDC_Miccai_lax2_15p.h5","ACDC_Miccai_lax2_15p_5fold.h5")
