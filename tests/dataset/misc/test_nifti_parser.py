import os

from preprocess.misc.nifti_parser import NiftiParser


class TestNiftiParser:

    def setup_class(self):
        if os.path.isfile("./tests/dataset/misc/target/ACDC_train.h5"):
            os.remove("./tests/dataset/misc/target/ACDC_train.h5")
            os.remove("./tests/dataset/misc/target/ACDC_test.h5")

    def test_read_all_nifti_files(self):
        parser = NiftiParser("tests/dataset/misc/data", "tests/dataset/misc/target", "ACDC_train", "ACDC_test")
        files_list, num = parser.read_all_nifti_files()
        assert 'patient099' in files_list
        assert 'patient100' in files_list
        assert 'patient099/01/label' in files_list['patient099']
        assert num == 2
        parser.save_nifti_to_hdf5(files_list, num)
