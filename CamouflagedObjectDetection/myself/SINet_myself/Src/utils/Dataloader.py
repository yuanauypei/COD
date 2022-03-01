import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from torchvision import utils as vutils
import sys
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from typing import List
import numbers
from collections.abc import Sequence
from torch import Tensor

# new adding-------starting
import math
import random
# from PIL import Image
import torch
# from torchvision import transforms as T
from torchvision.transforms import functional as F
import pdb
 
 

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.flip_prob = prob
 
    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self,  prob=0.5, keys=None):
        # super(RandomVerticalFlip, self).__init__(keys)
        assert 0 <= prob <= 1, "probability must be between 0 and 1"
        self.prob = prob
 
    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            if target is not None:
                target = F.vflip(target)
        return image, target

class RandomRotation(torch.nn.Module):

    def __init__(
        self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        fill_tar = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        if isinstance(target, Tensor):
            if isinstance(fill_tar, (int, float)):
                fill_tar = [float(fill_tar)] * F.get_image_num_channels(target)
            else:
                fill_tar = [float(f) for f in fill_tar]
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, fill), F.rotate(target, angle, self.resample, self.expand, self.center, fill_tar)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += ")"
        return format_string
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
        # return {'image':image, 'mask':mask}

def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")

def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

 



# new adding-------ending
import imageio
import torch.nn.functional as F2
import matplotlib
from PIL import Image



class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        # self.aug_transform = transforms.Compose([
        #     # transforms.RandomRotation(degrees=45, expand=False),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5)
        # ])
        # self.aug_transform = Compose([
        #     RandomRotation(degrees=45, expand=False),
        #     RandomHorizontalFlip(prob=0.5),
        #     RandomVerticalFlip(prob=0.5)
        # ])
        # self.count = 0

    def __getitem__(self, index):
        '''
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt
        '''

        #疑问：顺序问题
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # pdb.set_trace()
        # image = self.aug_transform(image)
        # image = self.img_transform(image)
        # gt = self.aug_transform(gt)
        # gt = self.gt_transform(gt)
        # image.save("/home/hebichang/CamDetector/model/SINet-master/pics/"+str(index)+"_ori.jpg")
        # gt.save("/home/hebichang/CamDetector/model/SINet-master/pics/"+str(index)+"_ori_gt.jpg")

        # image, gt = self.aug_transform(image, gt)
        # print(image, gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
    
        
        # a = torch.tensor([1.])
        # # print(a.type)
        # print(image.size())
        # print(gt.size())
        # image2 = np.array(image)
        # image2 = image.numpy()
        # image2 = Image.fromarray(image2)
        #used:
        # image.save("/home/hebichang/CamDetector/model/SINet-master/pics/"+str(index)+".jpg")
        # gt.save("/home/hebichang/CamDetector/model/SINet-master/pics/"+str(index)+"_gt.jpg")
        # vutils.save_image(image, "/home/hebichang/CamDetector/model/SINet-master/pics/"+str(index)+".jpg")
        # vutils.save_image(gt, "/home/hebichang/CamDetector/model/SINet-master/pics/"+str(index)+"_gt.jpg")
        # sys.exit(1)
        

        
        # pdb.set_trace()
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        # https://blog.csdn.net/Karen_Yu_/article/details/115293733
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            # img_arr = np.array(img)
            # print(img_arr.shape)
            

            # print("rgb:", f)
            # print(img)
            
            # arr1 = img_arr[:]
            # # print(arr1.shape)
            # arr2 = arr1.copy()
            # for x in range(1,arr1.shape[0]):
            #     for y in range(1,arr1.shape[1]):
            #         a = img_arr[x,y][0]
            #         b = img_arr[x,y][1]
            #         c = img_arr[x,y][2]
            #         arr1[x,y] = (0,0,c)
            # print("img_arr: ", img_arr)
            # print("arr1: ", arr1)
            # arr2 = arr2.transpose(2,0,1)
            # arr2[0,:,:]=0
            # arr2[1,:,:]=0
            # arr2 = arr2.transpose(1,2,0)
            # print(arr2==arr1)



            # image_last = Image.fromarray(img_arr)
            
            # print("")
            # image_last.show()
            
            # print(img_arr)
            
            # print(img_arr.shape)
            # print(arr1.shape)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # print("gt:", f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)

        img_name_list = self.images[index]

        return images, img_name_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    # `num_workers=0` for more stable training
    dataset = CamObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
