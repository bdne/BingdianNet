import cv2
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
class UnetDataset(Dataset):
    def __init__(self,img_ids,img_dir,mask_dir,img_ext,mask_ext,num_classes,transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
    def __len__(self):
        return len(self.img_ids)
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
