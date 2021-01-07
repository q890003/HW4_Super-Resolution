import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import architecture
from data import DIV2K
import utils

import random
import argparse, os
import sys
import cv2
import math
import numpy as np
import skimage.color as sc
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image

args = {}
args["batch_size"] = 16
args["testBatchSize"] = 1
args["nEpochs"] = 5000
args["lr"] = 2e-4
args["step_size"] = 1000
args["gamma"] = 0.5
args["cuda"] = True
args["resume"] = "path"
args["start_epoch"] = 1
args["threads"] = 1
args["root"] = "./data/training_hr_images/"
args["n_train"] = 10
args["n_val"] = 1
args["test_every"] = 2
args["scale"] = 3
args["patch_size"] = 96
args["rgb_range"] = 1
args["n_colors"] = 3
args["pretrained"] = "./checkpoint_x3/_test_epoch_5321_psnr25.25416464562951.pth"
args["seed"] = 47
args["isY"] = True
args["ext"] = ".npy"
args["phase"] = "train"


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1 / torch.sqrt(mse))


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


torch.backends.cudnn.benchmark = True
# random seed
seed = args["seed"]
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args["cuda"]
device = torch.device("cuda:0" if cuda else "cpu")

print("===> Loading datasets")

train_dataset = DIV2K.div2k(args)
# testset = Set5_val.DatasetFromFolderVal("../data/testing_lr_images/",
#                                        "Test_Datasets/Set5_LR/x{}/".format(scale),
#                                        scale)


test_dataset = DIV2K.div2k_test(args)

training_data_loader = DataLoader(
    dataset=train_dataset,
    num_workers=args["threads"],
    batch_size=args["batch_size"],
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)
testing_data_loader = DataLoader(
    dataset=test_dataset, num_workers=1, batch_size=args["testBatchSize"], shuffle=False
)

print(len(train_dataset), len(test_dataset))
psnr = PSNR()
ssim = SSIM()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


print("===> Building models")
args["is_train"] = True

model = architecture.IMDN(upscale=args["scale"])
l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args["pretrained"]:

    if os.path.isfile(args["pretrained"]):
        print("===> loading models '{}'".format(args["pretrained"]))
        checkpoint = torch.load(args["pretrained"])
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if "module" in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args["pretrained"]))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args["lr"])


def train(epoch):
    model.train()
    utils.adjust_learning_rate(
        optimizer, epoch, args["step_size"], args["lr"], args["gamma"]
    )
    print("epoch =", epoch, "lr = ", optimizer.param_groups[0]["lr"])
    _psnr = 0
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args["cuda"]:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        sr_tensor = sr_tensor.clamp(0, 1)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1
        _psnr += psnr(sr_tensor, hr_tensor)
        loss_sr.backward()
        optimizer.step()
        if iteration % 10 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss_l1: {:.5f}, psnr={:3.5f}".format(
                    epoch + 5000,
                    iteration,
                    len(training_data_loader),
                    loss_l1.item(),
                    psnr(sr_tensor, hr_tensor),
                ),
                end=", ",
            )


def valid(epoch):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for i, (lr_tensor, hr_tensor) in enumerate(testing_data_loader):
        if args["cuda"]:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
        with torch.no_grad():
            pre = model(lr_tensor)

        #         sr_img = pre.clamp(0, 1)
        #         sr_img = utils.tensor2np(pre.cpu()[0])
        #         gt_img = hr_tensor.cpu()[0]

        sr_img = utils.tensor2np(pre.cpu()[0])
        gt_img = utils.tensor2np(hr_tensor.cpu()[0])
        crop_size = args["scale"]
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args["isY"] is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    avg_psnr /= len(testing_data_loader)
    avg_ssim /= len(testing_data_loader)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr, avg_ssim))

    global best_psnr
    if best_psnr < avg_psnr:
        best_psnr = avg_psnr
        dir_name = "results/result_epoch{}_psnr{}".format(epoch + 5000, str(avg_psnr))

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, (lr_tensor, hr_tensor) in enumerate(testing_data_loader):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            with torch.no_grad():
                pre = model(lr_tensor)
            save_image(pre[0], dir_name + "/{:02d}.png".format(i))
        save_checkpoint(epoch + 5000, avg_psnr)


def save_checkpoint(epoch, avg_psnr):
    model_folder = "checkpoint_x{}/".format(args["scale"])
    model_out_path = model_folder + "_test_epoch_{}_psnr{}.pth".format(epoch, avg_psnr)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


print("===> Training")
print_network(model)
with torch.cuda.device(0):
    best_psnr = 24
    for epoch in range(args["start_epoch"], args["nEpochs"] + 1):
        valid(epoch)
        train(epoch)
