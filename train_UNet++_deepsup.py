import torch
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from Net.UNet7plus import *
from Utils.Dataset import Three_Fold
from Utils.Eval import UNetdice

from torch.optim import lr_scheduler
from Utils.Loss import LossFunction

import random

class Segmentation(object):
    def __init__(self):
        #图像尺寸
        self.width = 512
        self.height = 512
        #训练参数
        self.batch_size = 1
        self.test_batch_size = 2
        self.epochs = 30
        self.lr = 0.001
        self.min_lr = 1e-5
        self.num_classes = 2
        self.log_interval = 12
        self.save_folder='D:\\python\\BingdianNet\\result\\Liver2017\\'
        self.save_model_name='UNet++deepsup7_1.pth'
        #定义模型
        self.cuda = True
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model=NestedUNet(3,1,'True')
        print(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler=lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.min_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loss = LossFunction()
        #评价指标初始化
        self.best_dice = 0.0

         #创建模型存储文件夹
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # 训练数据读取
        train_dataset = Three_Fold(txt_path='D:\\python\\BingdianNet\\preprocess\\3fold.txt',
                                   mode='train', val_folder='train_1')
        # 验证数据读取
        val_dataset = Three_Fold(txt_path='D:\\python\\BingdianNet\\preprocess\\3fold.txt',
                                 mode='val', val_folder='train_1')
        #加载数据
        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, **kwargs)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.test_batch_size, shuffle=False, **kwargs)

    def train(self,epoch):
        dice_liver = 0
        self.model.train()
        for batch_idx, sample in enumerate(self.train_loader):
            # print(batch_idx)
            self.optimizer.zero_grad()
            img = sample['image'].cuda()
            target = sample['label'].cuda().long()
            outputs = self.model(img)
            loss = 0
            for output in outputs:
                loss += self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            #dice系数
            target_liver = torch.where(target == 1, torch.full_like(target, 1),
                                       torch.full_like(target, 0)).cuda().long()
            output_argmax = torch.argmax(outputs[-1], dim=1, keepdim=True)  # 预测值，keepdim能够保持当前维度
            output_liver = torch.where(output_argmax == 1, torch.full_like(output_argmax, 1),
                                       torch.full_like(output_argmax, 0)).cuda().long()
            liver = UNetdice.dice_coeff(output_liver, target_liver.unsqueeze(dim=1))
            dice_liver += liver

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Dice: {:.3f}'.format(
                    epoch, batch_idx * len(img), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item() ,100. * liver))

        dice_liver_acc = 100. * dice_liver / len(self.train_loader)
        print('训练集dice*****************************', dice_liver_acc)

    def test(self):

        self.model.eval()
        test_loss = 0
        dice_liver = 0
        dice_tumor = 0
        print('---------------------------------开始测试----------------------------')
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['image'].cuda()
                target = sample['label'].cuda().long()
                target_liver = torch.where(target == 1,torch.full_like(target, 1), torch.full_like(target, 0)).cuda().long()
                target_tumor = torch.where(target == 2,torch.full_like(target, 1), torch.full_like(target, 0)).cuda().long()
                outputs = self.model(img)
                test_loss = self.criterion(outputs[-1],target.squeeze(dim=1))
                output_argmax = torch.argmax(outputs[-1], dim=1, keepdim=True)#预测值，keepdim能够保持当前维度

                print('=1*****************',torch.sum(output_argmax==1))
                print('=2*****************',torch.sum(output_argmax==2))


                output_liver = torch.where(output_argmax == 1, torch.full_like(output_argmax, 1),
                                           torch.full_like(output_argmax, 0)).cuda().long()
                output_tumor = torch.where(output_argmax == 2, torch.full_like(output_argmax, 1),
                                           torch.full_like(output_argmax, 0)).cuda().long()

                liver = UNetdice.dice_coeff(output_liver, target_liver.unsqueeze(dim=1))
                #dice_liver += UNetdice.dice_coeff(output_liver, target_liver.unsqueeze(dim=1))
                dice_liver += liver

                tumor =UNetdice.dice_coeff(output_tumor, target_tumor.unsqueeze(dim=1))
                #dice_tumor += UNetdice.dice_coeff(output_tumor, target_tumor.unsqueeze(dim=1))
                dice_tumor += tumor
                print('肝脏', liver)
                print('肿瘤', tumor)

        test_loss /= len(self.val_loader.dataset)
        dice_liver_acc = 100. * dice_liver / len(self.val_loader)
        dice_tumor_acc = 100. * dice_tumor / len(self.val_loader)

        print('\nTest set: Batch average loss: {:.4f}, Dice liver Coefficient: {:.2f}% , Dice tumor Coefficient : {:.2f}% \n'
              .format(test_loss, dice_liver_acc, dice_tumor_acc))

        if dice_liver_acc > self.best_dice:
            torch.save(self.model.state_dict(), self.save_folder + self.save_model_name)
            self.best_dice = dice_liver_acc
            print("======Saving model======")


    def main(self):
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
        # 设置随机数种子

        setup_seed(123)
        for epoch in range(1, self.epochs + 1):
            print("学习率:",self.optimizer.param_groups[0]['lr'])
            Segmentation.train(self,epoch)
            self.scheduler.step()
            Segmentation.test(self)

seg=Segmentation()
seg.main()