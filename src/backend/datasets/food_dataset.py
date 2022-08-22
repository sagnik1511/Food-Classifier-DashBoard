import numpy as np
import pandas as pd
from PIL import Image
from os import path as osp
from torchvision.transforms import transforms


class FoodDataset:

    def __init__(self, root_dir, meta_df_path, h=224, w=224, channels=3, transform=None):
        self.root_dir = root_dir
        self.meta_df = pd.read_csv(osp.join(root_dir, meta_df_path))
        self.img_dir = osp.join(root_dir, meta_df_path.split(".")[0])
        self.h = h
        self.w = w
        self.channels = channels
        self.transform = self._generate_transformations(transform)

    def __len__(self):
        return len(self.meta_df)

    @staticmethod
    def _generate_transformations(transform):
        if transform:
            return transforms
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]
        )


    def _load_image(self, path):
        image = Image.open(osp.join(self.img_dir, path))
        image = image.resize((self.h, self.w))
        assert np.array(image).shape[2] == 3
        return image

    def __getitem__(self, index):
        path = self.meta_df.loc[index, "path"]
        image = self._load_image(path)
        label = self.meta_df.loc[index, "label"]
        image = self.transform(image)
        return image, label

