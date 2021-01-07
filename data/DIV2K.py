import torch.utils.data as data
import os.path
from PIL import Image
import numpy as np
from data import common
import torchvision.transforms as transforms


def default_loader(path):
    return Image.open(path)


def npy_loader(path):
    return np.load(path)


IMG_EXTENSIONS = [
    ".png",
    ".npy",
]


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class div2k(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt["scale"]
        self.root = self.opt["root"]
        self.ext = ".png"  # self.opt["ext"]   # '.png' or '.npy'(default)
        self.train = True if self.opt["phase"] == "train" else False
        self._set_filesystem(self.root)
        self.images_hr = make_dataset(self.dir_hr)
        print(len(self.images_hr))

        self.trans_hr = transforms.Compose(
            [
                transforms.RandomCrop((192, 192), pad_if_needed=True, fill=0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
            ]
        )
        self.trans_resizeFromhr = [  # Filters comparison table
            # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters
            transforms.Resize((64, 64), interpolation=0),  # NEAREST = NONE = 0
            transforms.Resize((64, 64), interpolation=1),  # LANCZOS = ANTIALIAS = 1
            transforms.Resize((64, 64), interpolation=2),  # BILINEAR = LINEAR = 2
            transforms.Resize((64, 64), interpolation=3),  # BICUBIC = CUBIC = 3
            transforms.Resize((64, 64), interpolation=4),  # BOX = 4
            transforms.Resize((64, 64), interpolation=5),  # HAMMING = 5
        ]

        self.trans_lr = transforms.Compose(
            [
                transforms.RandomChoice(self.trans_resizeFromhr),
                transforms.ToTensor(),
            ]
        )

    def _set_filesystem(self, dir_data):
        self.root = dir_data
        self.dir_hr = os.path.join(self.root)

    def __getitem__(self, idx):
        hr = Image.open(self.images_hr[idx])
        hr_w, hr_h = hr.size
        hr = self.trans_hr(hr)
        lr_tensor = self.trans_lr(hr)
        hr_tensor = transforms.ToTensor()(hr)
        return lr_tensor, hr_tensor

    def __len__(self):
        if self.train:
            return len(self.images_hr)


class div2k_test(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = "./data/"
        self.ext = ".png"  # self.opt["ext"]   # '.png' or '.npy'(default)
        self._set_filesystem(self.root)
        self.images_hr = [self.dir_hr + "{:02d}.png".format(i) for i in range(14)]
        self.images_lr = [self.dir_lr + "{:02d}.png".format(i) for i in range(14)]
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _set_filesystem(self, dir_data):
        self.root = dir_data
        self.dir_hr = os.path.join(self.root + "Set14/")
        self.dir_lr = os.path.join(self.root + "testing_lr_images/")

    def __getitem__(self, idx):
        hr = Image.open(self.images_hr[idx]).convert("RGB")
        lr = Image.open(self.images_lr[idx])

        lr_tensor = self.trans(lr)
        _, lr_h, lr_w = lr_tensor.size()
        hr_tensor = transforms.Resize((lr_h * 3, lr_w * 3), interpolation=3)(hr)
        hr_tensor = self.trans(hr_tensor)

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.images_hr)


from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    ToTensor,
    ToPILImage,
    CenterCrop,
    Resize,
    Normalize,
)


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    )


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose(
        [
            RandomCrop(crop_size, pad_if_needed=True),
            ToTensor(),
        ]
    )


def train_lr_transform(crop_size, upscale_factor):
    return Compose(
        [
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )


def display_transform():
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [
            join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)
        ]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


# class div2k_test(Dataset):
#     def __init__(self, opt):
#         super(div2k_test, self).__init__()
#         self.upscale_factor = 3
#         self.opt = opt
#         self.root = "../data/"
#         self.ext = ".png" #self.opt["ext"]   # '.png' or '.npy'(default)
#         self._set_filesystem(self.root)
#         self.image_filenames = make_dataset(self.dir_hr)
#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index]).convert('RGB')
#         h, w = hr_image.size
#         w_crop_size = calculate_valid_crop_size(w, self.upscale_factor)
#         h_crop_size = calculate_valid_crop_size(h, self.upscale_factor)
#         lr_scale = transforms.Resize((w_crop_size // self.upscale_factor, h_crop_size// self.upscale_factor), interpolation=Image.BICUBIC)
#         hr_image = transforms.Resize((w_crop_size, h_crop_size), interpolation=Image.BICUBIC)(hr_image)
# #         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_image)
#     def _set_filesystem(self, dir_data):
#         self.root = dir_data
#         self.dir_hr = os.path.join(self.root + "Set14/")
#         self.dir_lr = os.path.join(self.root + "testing_lr_images/")
#     def __len__(self):
#         return len(self.image_filenames)
