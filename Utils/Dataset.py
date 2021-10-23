import cv2
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
class UnetDataset(Dataset):
    def __init__(self,img_root=None,img_list=None,transform=None):
        self.img_root=img_root
        self.img_list=img_list
        self.transform = transform
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        image_name=self.img_list[index][:-4]
        if self.transform is not None:
            image = self.transform(image)
        return image,image_name