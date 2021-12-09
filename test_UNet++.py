import torch
import cv2
from Net.UNetplus import NestedUNet
from Utils.Dataset import UNet_Data
import os
import torch.nn as nn
import numpy as np


class Segmentation(object):
    def __init__(self):
        self.test_batch_size = 1
        self.cuda = True
        self.save_folder = '/home/sqy/disk/lsc_unet/BingdianNet/result/Liver2017/'
        self.save_model_name = 'Unet++_seed123_epoch=30_all.pth'

        ori_img_path = "D:\\python\\BingdianNet\\raw\\ISBI_2017_liver\\test"
        test_img_list = sorted(os.listdir(ori_img_path))

        # 测试数据读取
        test_dataset = UNet_Data(img_root=ori_img_path, img_list=test_img_list, mode='test')

        # 加载数据
        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                                       **kwargs)

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model = NestedUNet(3, 1)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load('.\\result\\Liver2017\\Unet++_seed123_epoch=30_all.pth'))

        self.pic = []

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, name) in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                # print(output.shape)
                self.pic.append((output, name))
        return self.pic

    def main(self):
        Segmentation.test(self)
        i = 0
        for img_tensor in self.pic:
            img = img_tensor[0].squeeze(0).permute(1, 2, 0).cpu()
            # print(img.shape)
            # img = Image.fromarray(img)
            img = torch.argmax(img, dim=2, keepdim=True)
            img = torch.where((img != 1) & (img != 2), torch.full_like(img, 0),
                                       img)
            img = torch.where(img == 1, torch.full_like(img, 128),img)
            img = torch.where(img == 2, torch.full_like(img, 255),img)
            img = img.numpy()
            print(img.shape)
            cv2.imwrite(".\\output\\liver_2017\\%s.jpg" % (img_tensor[1]), img)
            print('ok')
            i = i + 1


import argparse
import torch.distributed as dist


seg_UNet = Segmentation()
seg_UNet.main()