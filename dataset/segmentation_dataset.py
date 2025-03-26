import os.path
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from dataset.utils import get_transform


class SegmentationDataset(Dataset):
    def __init__(self, config, path, is_train=True):
        super(SegmentationDataset, self).__init__()
        self.is_train = is_train
        self.bright_image_path = os.path.join(path, "bright_images")
        self.image_path = os.path.join(path, "images")
        self.mask_path = os.path.join(path, "masks")
        self.transform = get_transform(config, is_train)
        self.image_name_list = os.listdir(self.mask_path)
        self.use_bright = config.use_bright
        assert len(self.image_name_list) == len(os.listdir(self.mask_path))

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        if self.use_bright and random.uniform(0, 1) < 0.2 and self.is_train:
            image_path = os.path.join(self.bright_image_path, image_name)
        else:
            image_path = os.path.join(self.image_path, image_name)
        mask_path = os.path.join(self.mask_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
        image = np.float32((image - mean) / std)
        mask = np.float32(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128)
        pair = self.transform(image=image, mask=mask)
        return pair['image'], pair['mask']