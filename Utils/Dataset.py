import cv2
from PIL import Image
import os
from torch.utils.data.dataset import Dataset

from Utils.Transform import Transform
class UNet_Data(Dataset):
    def __init__(self,img_root=None,label_root=None,img_list=None,label_list=
    None,mode=None):
        self.img_root=img_root
        self.img_list=img_list
        self.label_list=label_list
        self.label_root=label_root
        if mode=='train':
            self.transform = Transform('train') # 根据任务不同init函数传参不同
        elif mode == 'test':
            self.transform = Transform('test')
        elif mode == 'val':
            self.transform = Transform('val')
        else:
            print('训练模式错误')
            assert 1==2
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_root, self.img_list[index]))
        if self.label_list is not None:
            label = Image.open(os.path.join(self.label_root, self.label_list[index]))
            sample = {"image": image, "label": label}
            sample = self.transform(sample)
            return sample
        else:
            image_name = self.img_list[index][:-4]
            image = self.transform(image)
        return image, image_name



class Three_Fold(Dataset):
    def __init__(self, txt_path=None, mode=None, val_folder=None):
        self.all_data_info = self.get_img_info(txt_path)
        self.root_path = 'D:\\python\\BingdianNet\\raw\\ISBI_2017_liver\\3fold'
        if mode == 'train':
            self.data_info = [element for element in self.all_data_info if (element[1] != val_folder)]
            self.transform = Transform('train')  # 根据任务不同init函数传参不同
        elif mode == 'val':
            self.data_info = [element for element in self.all_data_info if (element[1] == val_folder)]
            self.transform = Transform('val')
        else:
            print('模式错误')
            assert 1==2

    def __getitem__(self, index):
        # Dataset读取图片的函数
        img_name, folder = self.data_info[index]
        label_folder = 'label_'+folder[-1]
        img_path = os.path.join(self.root_path,folder,img_name)
        label_path = os.path.join(self.root_path,label_folder,img_name)
        image = Image.open(img_path)
        label = Image.open(label_path)
        sample = {"image": image, "label": label}
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(txt_path):
        data_info = []
        data = open(txt_path, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.split('@')
            img_name = data_line[0]
            folder = data_line[1][:-1]
            # path = os.path.join(root_path,folder)
            data_info.append((img_name,folder))
        return data_info
# from Transform import NiiTransform
# import SimpleITK as sitk
# class CreateNiiDataset(Dataset):
#     def __init__(self,img_root=None,label_root=None,img_list=None,label_list=
#     None,mode=None):
#         self.img_root=img_root
#         self.img_list=img_list
#         self.label_list=label_list
#         self.label_root=label_root
#         if mode=='train':
#             self.transform = NiiTransform('train') # 根据任务不同init函数传参不同
#         elif mode == 'test':
#             self.transform = NiiTransform('test')
#         elif mode == 'val':
#             self.transform = NiiTransform('val')
#         else:
#             print('训练模式错误')
#             assert 1==2
#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, index):
#         image = sitk.ReadImage(os.path.join(self.img_root, self.img_list[index]))
#         if self.label_list is not None:
#             label = sitk.ReadImage(os.path.join(self.label_root, self.label_list[index]))
#             sample = {"image": image, "label": label}
#             sample = self.transform(sample)
#             return sample
#         else:
#             image_name = self.img_list[index][:-4]
#             image = self.transform(image)
#         return image, image_name







# import numpy as np
# class UNetplusDataset(Dataset):
#     def __init__(self,img_root=None,label_root=None,img_list=None,transform=None,target_transform=None):
#         self.img_root=img_root
#         self.img_list=img_list
#         self.transform = transform
  
#         self.label_root=label_root
#         self.target_transform = target_transform
        
#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, index):
#         image = Image.open(os.path.join(self.img_root, self.img_list[index]))

       
#         if self.transform is not None:
#             image = self.transform(image)
#             # print(image.shape)
#             # print(image.mode)
#             # print(np.array(image).shape)
#             # assert 1==2

#         label = Image.open(os.path.join(self.label_root, self.img_list[index]))
#         if self.target_transform is not None:
#             label = self.target_transform(label)
        
#         return image, label



import torch
if __name__ == '__main__':
#存储图片id，区分训练集与验证集
    cuda = True
    batch_size = 2
    test_batch_size = 2 
    ori_img_path = "D:\\traindata\\ISBI_2017_liver\\processed"
    ori_list = sorted(os.listdir(ori_img_path))
    label_path = "D:\\traindata\\ISBI_2017_liver\\label"
    label_list = sorted(os.listdir(label_path))
    train_img_list, val_img_list = ori_list[:106], ori_list[106:]
    train_label_list, val_label_list = label_list[:106], label_list[106:]

    #训练数据读取
    train_dataset = CreateNiiDataset(img_root=ori_img_path,label_root=label_path,img_list=train_img_list,label_list=train_label_list, mode='train')
    #验证数据读取
    val_dataset = CreateNiiDataset(img_root=ori_img_path,label_root=label_path,img_list=val_img_list,label_list=val_label_list, mode='train')
    
    #加载数据
    kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=test_batch_size, shuffle=False, **kwargs)
    device = torch.device("cuda" if cuda else "cpu")
    

    for batch_idx, sample in enumerate(train_loader):
        # print(sample['label'].shape)
        print('end')