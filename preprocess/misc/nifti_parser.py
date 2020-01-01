"""
NiftiParser
Read Nifti format datasource and convert the files as a hdf5 format.
## GPL License
## Author: Qi Chang<qc58@cs.rutgers.edu>
"""
import importlib
import os
import re
import stringcase
import h5py
import nibabel as nib

from utils import util


class NiftiParser:
    def __init__(self, source_path: str, save_path: str, training_file_name: str, testing_file_name: str
                 , split_ratio: float = 0.2, convert_type: str = "hdf5",name_handler: str = 'common_name_handler'):
        """
        Initial data related information.
        :param source_path: source path
        :param save_path: target path
        :param training_file_name: training file name, will delete '.h5' if exist, and append '_001'...'_999' if file
        number exceed one_file_limit.
        :param testing_file_name: same as training file name. if training_file_name same as testing_file_name, then will
        append to the same file.
        :param convert_type: current only option is "hdf5"
        :param split_ratio: split ratio for testing 0.2
        :param name_handler: customized file name parser. Because the keys in the target file are based on different
        Nifti name patterns or other reference files, you need to customize how to store the Nifti file in H5.
        """
        self.source_path = source_path
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.training_file_name = training_file_name.replace(".h5", "")
        self.testing_file_name = testing_file_name.replace(".h5", "")
        self.convert_type = convert_type
        self.split_ratio = split_ratio

        module_ = importlib.import_module("preprocess.misc." + name_handler)

        self.name_handler = getattr(module_, stringcase.pascalcase(name_handler))

        self.acceptable_file_format = re.compile(r'\.nii\.gz$')

    def read_all_nifti_files(self):
        all_nifti_file = {}
        folder_number = 0
        for root, subdirs, files in os.walk(self.source_path):
            print(f'root:{root}, subdirs:{subdirs}, files:{files}')
            for file in files:
                print(f"read file {file}")
                if self.acceptable_file_format.search(file):
                    parent_folder, label = self.name_handler.parse_name(root, file)
                    # nifti_file = nib.load(os.path.join(root, file))
                    # array_file = nifti_file.get_fdata()
                    # if parent_folder not in all_nifti_file:
                    #     all_nifti_file[parent_folder] = {}
                    #     folder_number += 1
                    # all_nifti_file[parent_folder][label] = array_file
                    # all_nifti_file[parent_folder][label + '$$pixdim'] = nifti_file.header.get('pixdim')
                    # all_nifti_file[parent_folder][label + '$$dim'] = nifti_file.header.get('dim')

                    if parent_folder not in all_nifti_file:
                        all_nifti_file[parent_folder] = {}
                        folder_number += 1
                    all_nifti_file[parent_folder][label] = os.path.join(root, file)
        return all_nifti_file, folder_number

    def save_nifti_to_hdf5(self, all_nifti_file, folder_number):
        save_training_file = h5py.File(os.path.join(self.save_path, self.training_file_name + ".h5"), 'w')
        if self.training_file_name != self.testing_file_name:
            save_testing_file = h5py.File(os.path.join(self.save_path, self.testing_file_name + ".h5"), 'w')
        else:
            save_testing_file = save_training_file

        training_number = round(folder_number * (1 - self.split_ratio))
        cur_idx = 0
        for folder in all_nifti_file:
            print(f"Save data: {folder}")
            files_in_folder = all_nifti_file[folder]
            if cur_idx < training_number:
                d_type = "train"
                current_file = save_training_file
            else:
                d_type = "test"
                current_file = save_testing_file

            for file_name in files_in_folder:
                nifti_file = nib.load(files_in_folder[file_name])
                array_file = nifti_file.get_fdata()
                pixdim = nifti_file.header.get('pixdim')
                dim = nifti_file.header.get('dim')

                ds = current_file.create_dataset(f"{d_type}/{file_name}", data=array_file)
                ds.attrs["x_spacing"] = pixdim[1]
                ds.attrs["y_spacing"] = pixdim[2]
                ds.attrs["z_spacing"] = pixdim[3]

                ds.attrs['dim'] = dim

            cur_idx += 1

        save_training_file.close()
        if self.training_file_name != self.testing_file_name:
            save_testing_file.close()


if __name__ == "__main__":
    # ACDC Parser
    parser = NiftiParser("/Users/qichang/Downloads/training", "/Users/qichang/PycharmProjects/medical_dataset/ACDC_new",
                         "ACDC_train", "ACDC_train", split_ratio=0)

    # Brats Parser
    # parser = NiftiParser("/Users/qichang/PycharmProjects/gar_v0.1/data/BRATS/MICCAI_BraTS_2018_Data_Training/LGG",
    #                      "/Users/qichang/PycharmProjects/medical_dataset/BraTS2018",
    #                      "BraTS18_LGG_train", "BraTS18_LGG_test", name_handler="brats18_name_handler")

    all_nifti_files, folder_number = parser.read_all_nifti_files()
    parser.save_nifti_to_hdf5(all_nifti_files, folder_number)
