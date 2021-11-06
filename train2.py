import torch 
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from Net.UNet import UNet
from Utils.Dataset import UNet_Data
from Utils.Eval import UNetdice
class Segmentation(object):
    def __init__(self):
        #训练参数
        self.batch_size=1
        self.test_batch_size=2
        self.epochs=100
        self.lr=0.01
        self.cuda=True
        self.num_classes=2
        self.log_interval=3
        self.save_folder='D:\\python\\BingdianNet\\result\\'
        self.save_model_name='my_unet_classification.pth'
        #定义模型
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model=UNet(1,2)#n_classes=2 前景概率和背景概率
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
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
        train_dataset = UNet_Data(img_root=ori_img_path,label_root=label_path,img_list=train_img_list,label_list=train_label_list)
        #验证数据读取
        val_dataset = UNet_Data(img_root=ori_img_path,label_root=label_path,img_list=val_img_list,label_list=val_label_list)
        
        #加载数据
        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.test_batch_size, shuffle=False, **kwargs)

    def train(self,epoch):
        self.model.train()
        dice = 0
        for batch_idx, sample in enumerate(self.train_loader):
            # print("data shape:{}".format(data.dtype))
            # print(batch_idx)
            self.optimizer.zero_grad()
            img = sample['image'].cuda()
            target = sample['label'].cuda().long()
            print('target:',target.shape)
            output = self.model(img)
            labels = torch.full(size=(1,self.num_classes,512,512), fill_value=0).cuda()
            labels.scatter_(dim=1, index=target, value=1)#one-hot编码
            log_prob = F.log_softmax(output, dim=1)#logsoftmax
            loss = -torch.sum(log_prob * labels) #交叉熵
            print('labels',labels.shape)
            print('output',output.shape)
            output_argmax=torch.argmax(output, dim=1, keepdim=False)#预测值
            print(output_argmax.shape)
            # dice += UNetdice.dice_coeff(m(output), labels)
            # print(dice)
            loss.backward()
            self.optimizer.step()
        #     if batch_idx % self.log_interval == 0:
        #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, batch_idx * len(img), len(self.train_loader.dataset),
        #             100. * batch_idx / len(self.train_loader), loss.item()))
        # print('loader',len(self.train_loader))
        # dice_acc = 100. * dice / len(self.train_loader)
        # print('Train Dice coefficient: {:.2f}%'.format(dice_acc))
seg=Segmentation()
seg.train(1)