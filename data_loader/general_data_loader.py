"""
General Data Loader and General Dataset
General purpose data loader and dataset.
## GPL License
## Author: Qi Chang<qc58@cs.rutgers.edu>
"""
import h5py
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class GeneralDataLoader(DataLoader):
    """
    General data loader, which try to cover the most of the dataset loader requirements,
    so you do not need to customize data loader and dataset.
    If you are interested in using this general dataloader directly,
    the HDF5 dataset must follow the format below:
    1. type 1 train/validation in one file: one hdf5 contains "train"/"val" two keys, which has all records for training and
    validation. In each record, has "data"/"label" indicates data(dimension: ch x h x w (xd)) and label .
    eg: data_dir="xxx.h5", validation_split=[["train"],["val"]]
    2. type 2 train/validation in separate hdf5 files: two hdf5 contains keys: "train" and "val" separately.
    eg: data_dir=["xxx_train.h5","xxx_val.h5"], validation_split=[["train"],["val"]]
    3. type 3 train/split validation in one file with percentage: one hdf5 contains "train" key, validation_split is a
    float number which indicates the percentage of splitting training dataset as train and validation set.
    eg: data_dir="xxx.h5", validation_split=0.2
    4. type 4 cross-validation in one file with subset datasets: one hdf5 contains multiple keys,
    for instance ["1","2","3","4","5"], and the validation_split is a two dimension array indicates which subsets are
    for training, which subsets are for testing.
    eg: data_dir="xxx.h5", validation_split=[["1","2","3","4"],["5"]]
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 training=True, transforms=None):
        """
        :param data_dir: str|array: String should be the source of h5 file for GeneralDataset.
                                    Array should contain 2 string elements, contain the path to the training('train')
                                     and validation('val').
        :param batch_size:
        :param shuffle:
        :param validation_split: float|array: Float indicate split percentage of validation dataset.
                                              Array:[[/*training paths*/],[/*validation paths*/]].
                                                eg: [["train"],["val"]] or [["1","2","3","4"],["5"]]
        :param num_workers:
        :param collate_fn:
        """

        if isinstance(validation_split, list):
            assert len(validation_split) >= 2, "The validation_split array should contains at least 2 sub arrays for " \
                                               "train and validation"
            # Should load two h5 files which is training and validation dataset.
            if isinstance(data_dir, str):
                data_dir = [data_dir, data_dir]
            self.train_dataset = GeneralDataset(data_dir[0], transforms=transforms, paths=validation_split[0])
            self.valid_dataset = GeneralDataset(data_dir[1], transforms=transforms, paths=validation_split[1])

            print(f"GeneralDataLoader: training dataset length:{len(self.train_dataset)}, "
                  f"validation dataset length:{len(self.valid_dataset)}")

            self.validation_split = validation_split
            self.shuffle = shuffle

            self.init_kwargs = {
                'dataset': self.train_dataset,
                'batch_size': batch_size,
                'shuffle': self.shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super().__init__(**self.init_kwargs)
        elif isinstance(validation_split, float):
            # if data_dir is not list but string, also validation_split is float, split the dataset as val.
            assert isinstance(data_dir, str), "if validation_split is float, the data_dir only support string type"
            dataset = GeneralDataset(data_dir, transforms=transforms, paths=["train"])
            print(f"GeneralDataLoader: split dataset by percentage:{validation_split * 100}%")
            self.validation_split = validation_split
            self.shuffle = shuffle

            self.batch_idx = 0
            self.n_samples = len(dataset)

            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': self.shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        # TODO: need to add split as tuple which indicates
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if hasattr(self, 'valid_sampler'):
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        elif hasattr(self, 'valid_dataset'):
            val_init_kwargs = self.init_kwargs
            val_init_kwargs['dataset'] = self.valid_dataset
            return DataLoader(**self.init_kwargs)


class GeneralDataset(data.Dataset):
    def __init__(self, h5_filepath, transforms=None, paths=["train"]):
        """

        :param h5_filepath:string h5 file path to load.
        :param transforms: my_transforms
        :param paths: array of all keys which will loaded
        """
        super(GeneralDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.transforms = transforms
        self.img_list = self._get_img_list(h5_filepath, paths)

    def _get_img_list(self, h5_filepath, paths):
        img_list = []

        with h5py.File(h5_filepath, 'r') as h5_file:
            for path in paths:
                for key in h5_file[path].keys():
                    img_list.append((f"{path}/{key}/data", f"{path}/{key}/label"))
            h5_file.close()
        return img_list

    def __getitem__(self, index):
        with h5py.File(self.h5_filepath, 'r') as h5_file:
            img_path, label_path = self.img_list[index]
            img, label = h5_file[img_path][()], h5_file[label_path][()]

            if self.transforms is not None:
                img = self.transforms({'image': img, 'mask': label})

        return img['image'], img['mask']

    def __len__(self):
        return len(self.img_list)
