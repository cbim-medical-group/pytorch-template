
from base import BaseDataLoader
from data_loader import my_transforms
from dataset.acdc_dataset import ACDC_dataset


class AcdcDataLoader(BaseDataLoader):
    def __init__(self, data_root, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomHorizontalFlip(),
                # my_transforms.RandomRotation(10),
                my_transforms.RandomCrop(img_size, padding=40, fill_val=(0,0)),
                # my_transforms.LabelBinarization(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            trsfm = my_transforms.Compose([
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

        self.dataset = ACDC_dataset(data_root, training, transform=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
