import cv2
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
class UnetDataset(Dataset):
    def __init__(self,img_root=None,label_root=None,img_list=None,label_list=
    None,transform=None,target_transform=None):
        self.img_root=img_root
        self.img_list=img_list
        self.transform = transform
        self.label_list=label_list
        self.label_root=label_root
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        
       
        if self.transform is not None:
            image = self.transform(image)

       
        if self.label_list is not None:
            label = Image.open(os.path.join(self.label_root, self.label_list[index]))
            if self.target_transform is not None:
                label = self.target_transform(label)
            return image, label
        else:
            return image

class UnetTestDataset(Dataset):
    """
    You need to inherit nn.Module and
    overwrite __getitem__ and __len__ methods.
    """
    def __init__(self, img_root=None,
                 img_list=None, transform=None):

        self.img_root = img_root
        self.img_list = img_list
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        image_name = self.img_list[index][:-4]
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name

    def __len__(self):
        return len(self.img_list)

        
import numpy as np
class UNetplusDataset(Dataset):
    def __init__(self,img_root=None,label_root=None,img_list=None,transform=None,target_transform=None):
        self.img_root=img_root
        self.img_list=img_list
        self.transform = transform
  
        self.label_root=label_root
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))

       
        if self.transform is not None:
            image = self.transform(image)
            # print(image.shape)
            # print(image.mode)
            # print(np.array(image).shape)
            # assert 1==2

        label = Image.open(os.path.join(self.label_root, self.img_list[index]))
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label


