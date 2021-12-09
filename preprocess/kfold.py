import torch
import torch.nn as nn
from torch.utils.data.dataset import *
from PIL import Image
from torch.nn import functional as F
import random

class KZDataset(Dataset):
    def __init__(self, txt_path=None, ki=0, K=5, typ='train', transform=None, rand=False):
        '''
        txt_path: 所有数据的路径，我的形式为(单张图片路径 类别\n)
        	img1.png 0
        	...
     	    img100.png 1
     	ki：当前是第几折,从0开始，范围为[0, K)
     	K：总的折数
     	typ：用于区分训练集与验证集
     	transform：对图片的数据增强
     	rand：是否随机
        '''

        self.all_data_info = self.get_img_info(txt_path)
        
        if rand:
	        random.seed(1)
        	random.shuffle(self.all_data_info)
        leng = len(self.all_data_info)
        every_z_len = leng // K
        if typ == 'val':
            self.data_info = self.all_data_info[every_z_len * ki : every_z_len * (ki+1)]
        elif typ == 'train':
            self.data_info = self.all_data_info[: every_z_len * ki] + self.all_data_info[every_z_len * (ki+1) :]
            
        self.transform = transform

    def __getitem__(self, index):
    	# Dataset读取图片的函数
        img_pth, label = self.data_info[index]
        img = Image.open(img_pth).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(txt_path):
    	# 解析输入的txt的函数
    	# 转为二维list存储，每一维为 [ 图片路径，图片类别]
        data_info = []
        data = open(txt_path, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.split()
            img_pth = data_line[0]
            label = int(data_line[1])
            data_info.append((img_pth, label))
        return data_info   
