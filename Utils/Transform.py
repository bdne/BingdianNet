from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import random
import torchvision.transforms.functional as TF

class Transform(object):
    def __init__(self,mode):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        elif mode == 'val':
            self.mode = 'val'
        else:
            print('训练模式错误')
            assert 1==2
    def __call__(self,sample):
        if self.mode == 'train':
            return self._train_(sample)
        elif self.mode == 'test':
            return self._test_(sample)
        elif self.mode == 'val':
            return self._val_(sample)
        else:
            print('训练模式错误')
            assert 1==2
#'''''''''''''''''''''''''''''''''''''''''''
    def _train_(self,sample):
        sample = self.grayscale(sample)
        sample = self._random_rotate_(sample)
        sample = self._random_flip_(sample)
        sample = self.toBinary(sample)
        sample = self.ToTensor(sample)
        # sample = self.add_dimension(sample)
        return sample

    def _val_(self,sample):
        sample = self.grayscale(sample)
        sample = self.ToTensor(sample)
        return sample

    def _test_(self,img):
         img = self.test_grayscale(img)
         img = self.test_ToTensor(img)
         return img
# '''''''''''''''''''''''''''''''''''''''''''
    def test_grayscale(self,img):
        gray = transforms.Grayscale()
        img = gray(img)
        return img

    def test_ToTensor(self,img):
        tensor = transforms.ToTensor()
        img = tensor(img)
        return img
# ''''''''''''''''''''''''''''''''''''''''''''
# def ToNumpy(self)

    def ToTensor(self,sample):
        img = sample['image']
        label = sample['label']
        tensor = transforms.ToTensor()
        img = tensor(img)
        label = tensor(label)
        return {"image":img,"label":label}   

    def toBinary(self,sample):#对灰度图进行二值化处理
        img = sample['image']
        label = sample['label']
        label = np.array(label)
        ones = np.ones_like(label)*255
        binary_label = ones * (label > 127)
        binary_label = Image.fromarray(binary_label)
        # binary_label.show()
        # img.show()
        # assert 1==2
        return {'image':img,'label':binary_label}

    def grayscale(self,sample):
        img = sample['image']
        label = sample['label']
        gray = transforms.Grayscale()
        img = gray(img)
        label = gray(label)
        return {"image":img,"label":label}

    def _random_rotate_(self, sample, range_degree=90):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-range_degree,range_degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)#双线性插值
        mask = mask.rotate(rotate_degree, Image.NEAREST)#最近邻插值
        return {"image":img,"label":mask}

    def _random_flip_(self,sample):
        img = sample['image']
        mask = sample['label']
        temp = random.randint(0,2)
        img = {0:img, 1:TF.hflip(img), 2:TF.vflip(img)}[temp]
        mask = {0:mask, 1:TF.hflip(mask), 2:TF.vflip(mask)}[temp]

        return {"image":img,"label":mask}
      






