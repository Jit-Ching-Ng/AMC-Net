import cv2
import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class Deblur(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.3):
        super(Deblur, self).__init__(always_apply, p)

    def apply(self, img, **params):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)


def get_transform(config, is_train):
    transform_list = [A.Resize(config.image_size, config.image_size)]
    if is_train:
        transform_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            Deblur(p=0.3),
        ])
    transform_list.append(ToTensorV2())
    return A.Compose(transform_list, is_check_shapes=False)
