from dataset.acdc_test_dataset import AcdcTestDataset


class TestAcdcTestDataset:

    def test_prepare_data(self):
        ds = AcdcTestDataset("data", False, None, False, 224)
        assert len(ds) == len(ds.prepared_data_list)
        for i in range(len(ds.prepared_data_list)):
            assert ds.prepared_data_list[i]['img'].shape == (224, 224)
