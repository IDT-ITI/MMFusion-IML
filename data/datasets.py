"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
from torch.utils.data import Dataset
import random
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from contextlib import contextmanager


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def get_random_crop_coords_on_grid(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):

    y1 = int((height - crop_height + 1) * h_start)
    # MOVE CROP ON JPEG GRID
    y1 = y1 // 8 * 8

    y2 = y1 + crop_height

    x1 = int((width - crop_width + 1) * w_start)
    # MOVE CROP ON JPEG GRID
    x1 = x1 // 8 * 8

    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop_on_grid(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords_on_grid(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


class RandomCropONJPEGGRID(A.RandomCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(height, width, always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        return random_crop_on_grid(img, self.height, self.width, h_start, w_start)

    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}


class ManipulationDataset(Dataset):
    def __init__(self, path, image_size, train=False):
        self.train = train
        self.path = path
        self.image_size = image_size
        self.image_paths = []
        self.mask_paths = []
        self.base_path = './data'
        self.labels = []
        self.name = self.path.split('/')[-1].replace('.txt', '').replace('IDT-', '')

        self._init_transforms()
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.rstrip().split(' ')
                try:
                    image_path, mask_path, label_str = parts
                except ValueError:
                    print("Incorrect amount of columns in file, expected 4")
                    print(parts)

                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
                self.labels.append(int(label_str))

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return self.name

    def _init_transforms(self):

        self.image_transforms_train = A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),
            A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT, value=127, mask_value=-1, p=1),
            A.RandomCrop(height=self.image_size, width=self.image_size, p=1),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        ])

        self.image_transforms_final = A.Compose([
            ToTensorV2()
        ])

    def __getitem__(self, index):
        # ----------
        # Read image and label
        # ----------
        with cwd(self.base_path):
            image = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        label = self.labels[index]

        # ----------
        # Read mask
        # ----------
        if self.mask_paths[index] == 'None':
            mask = np.zeros((h, w), np.uint8)  # a totally black mask for real image
        else:
            with cwd(self.base_path):
                mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)



        if self.train:
            res = self.image_transforms_train(image=image, mask=mask)
            image = res['image']
            mask = res['mask']
        elif h > 2048 or w > 2048:
            res = A.LongestMaxSize(max_size=2048)(image=image, mask=mask)
            image = res['image']
            mask = res['mask']

        image = self.image_transforms_final(image=image)['image']
        image = image / 256.0

        mask = mask / 255.0
        mask = ToTensorV2()(image=mask)['image']
        mask = (mask > 0.1).long()

        return image, [], mask, label


class MixDataset(Dataset):
    def __init__(self, paths, image_size, train=False, class_weight=None):
        self.train = train
        self.paths = paths
        self.image_size = image_size
        self.dataset_list = []
        for path in paths:
            self.dataset_list.append(ManipulationDataset(path, image_size, train=train))

        if class_weight is None:
            self.class_weights = torch.FloatTensor([1.0, 1.0])
        else:
            self.class_weights = torch.FloatTensor(class_weight)

        self.lens = [len(d) for d in self.dataset_list]
        self.smallest = min(self.lens)

    # Should be shuffled every epoch to make sure different samples from big datasets are selected
    def shuffle(self):
        for dat in self.dataset_list:
            temp = list(zip(dat.image_paths, dat.mask_paths, dat.labels))
            random.shuffle(temp)
            image_paths, mask_paths, labels = zip(*temp)
            dat.image_paths, dat.mask_paths, dat.labels = list(image_paths), list(mask_paths), list(labels)

    def __len__(self):
        return len(self.dataset_list) * self.smallest

    def get_info(self):
        s = "Using datasets:\n"
        for ds in self.dataset_list:
            s += (str(ds)+' - '+str(len(ds))+' Images')
            s += '\n'
        s += "Data Configuration: "
        s += f"crop_size={self.image_size}, class_weight={self.class_weights}\n"
        return s

    def __getitem__(self, index):
        # balanced sampling
        if index < self.smallest * len(self.dataset_list):
            return self.dataset_list[index // self.smallest][index % self.smallest]
        else:
            raise ValueError("Something wrong.")

