import torch 
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from Net.UNet import UNet
from Utils.Dataset import UNet_Data
from Utils.Eval import UNetdice

from torch.optim import lr_scheduler
from Utils.Loss import LossFunction

class Segmentation(object):
    def __init__(self):
        #图像尺寸
        self.width = 512
        self.height = 512
        #训练参数
        self.batch_size=1
        self.test_batch_size=1
        self.epochs=100
        self.lr=0.001
        self.min_lr=1e-4
        self.num_classes=2
        self.log_interval=12
        self.save_folder='D:\\python\\BingdianNet\\result\\'
        self.save_model_name='my_unet_classification.pth'
        #定义模型
        self.cuda=True
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model=UNet(1,2)#n_classes=2 前景概率和背景概率
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

        #存储图片id，区分训练集与验证集
        ori_img_path = "D:\\python\\BingdianNet\\raw\\unet\\train"
        ori_list = sorted(os.listdir(ori_img_path))
        label_path = "D:\\python\\BingdianNet\\raw\\unet\\label"
        label_list = sorted(os.listdir(label_path))
        train_img_list, val_img_list = ori_list[:24], ori_list[24:]
        train_label_list, val_label_list = label_list[:24], label_list[24:]

        #训练数据读取
        train_dataset = UNet_Data(img_root=ori_img_path,label_root=label_path,img_list=train_img_list,label_list=train_label_list,mode='train')
        #验证数据读取
        val_dataset = UNet_Data(img_root=ori_img_path,label_root=label_path,img_list=val_img_list,label_list=val_label_list,mode='val')
        
        #加载数据
        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.test_batch_size, shuffle=False, **kwargs)

    def train(self,epoch):
        self.model.train()

        for batch_idx, sample in enumerate(self.train_loader):
            # print(batch_idx)
            self.optimizer.zero_grad()
            img = sample['image'].cuda()
            target = sample['label'].cuda().long()
            output = self.model(img)
            loss = self.loss(output,target.squeeze(dim=1))
            # loss = self.loss(output,target)#使用 one-hot的损失函数
            loss.backward()
            self.optimizer.step()
        
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(img), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self):

        self.model.eval()
        test_loss = 0
        dice = 0
        print('---------------------------------开始测试----------------------------')
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['image'].cuda()
                target = sample['label'].cuda().long()
                output = self.model(img)
                test_loss = self.criterion(output,target.squeeze(dim=1))
                output_argmax = torch.argmax(output, dim=1, keepdim=True)#预测值，keepdim能够保持当前维度
                dice += UNetdice.dice_coeff(target, output_argmax)

        test_loss /= len(self.val_loader.dataset)
        dice_acc = 100. * dice / len(self.val_loader)

        print('\nTest set: Batch average loss: {:.4f}, Dice Coefficient: {:.2f}%\n'.format(test_loss, dice_acc))

        if dice_acc > self.best_dice:
            torch.save(self.model.state_dict(), self.save_folder + 'UNet\\'+self.save_model_name)
            self.best_dice = dice_acc
            print("======Saving model======")



    def main(self):
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step()
            # print("学习率:",self.optimizer.param_groups[0]['lr'])
            Segmentation.train(self,epoch)
            Segmentation.test(self)

seg=Segmentation()
seg.main()