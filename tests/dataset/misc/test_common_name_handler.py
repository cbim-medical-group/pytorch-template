from preprocess.misc.common_name_handler import CommonNameHandler


class TestCommonNameHanlder:

    def test_parse_name(self):
        assert CommonNameHandler.parse_name("./data/patient100", "patient100_4d.nii.gz") == (
            "patient100", "patient100/data_without_label")
        assert CommonNameHandler.parse_name("./data/patient100", "patient100_frame13.nii.gz") == (
            "patient100", "patient100/13/data")
        assert CommonNameHandler.parse_name("./data/patient100", "patient100_frame13_gt.nii.gz") == (
            "patient100", "patient100/13/label")

        assert CommonNameHandler.parse_name("./data/patient100", "patient100_frame01.nii.gz") == (
            "patient100", "patient100/01/data")
        assert CommonNameHandler.parse_name("./data/patient100", "patient100_frame01_gt.nii.gz") == (
            "patient100", "patient100/01/label")
        assert CommonNameHandler.parse_name("./data/patient100", "patient100_frame001.nii.gz") == (
            "patient100", "patient100/001/data")
        assert CommonNameHandler.parse_name("./data/patient100", "patient100_frame001_gt.nii.gz") == (
            "patient100", "patient100/001/label")
