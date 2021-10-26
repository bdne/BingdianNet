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
        # print(np.array(image).shape)
        if self.transform is not None:
            image = self.transform(image)
            #print(image.shape)
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
    def __init__(self, img_root=None, label_root=None, img_list=None,
    label_list=None, transform=None, target_transform=None):
        self.img_root=img_root
        self.label_root=label_root
        self.img_list=img_list
        # self.label_list=label_list
        self.target_transform = target_transform
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        img_id = self.img_list[index]
        img = Image.open(os.path.join(self.img_root, img_id))
        img=np.array(img)
        print(type(img))
        assert 1==2
        #img = cv2.imread(os.path.join(self.img_root, img_id))

        #mask=cv2.imread(os.path.join(self.label_root,
                    #img_id), cv2.IMREAD_GRAYSCALE)[..., None]
        mask = Image.open(os.path.join(self.label_root,img_id))
        mask=np.array(mask)
        print('大小',(mask.shape))
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
