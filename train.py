import torch
from torchvision import transforms,models
from Net.UNet import UNet
from Utils.Dataset import UnetDataset
from Utils.Transform import toBinary
from Utils.Eval import UNetdice
import os
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from Utils.Eval import iou_score
class Parser(object):
        def __init__(self,parser):
            self.batch_size=parser['batch_size']
            self.test_batch_size=parser['test_batch_size']
            self.epochs=parser['epochs']
            self.lr=parser['lr']
            self.cuda=parser['cuda']
            self.save_folder=parser['save_folder']
            self.log_interval=parser['log_interval']
            self.save_model_name=parser['save_model_name']
        def get_item(self):
            print(self.epochs) 


class Segmentation(object):
    def __init__(self):

        #实例化参数
        self.parser_dict={}#定义一个参数字典
        self.parser_dict['batch_size']=2
        self.parser_dict['test_batch_size']=1
        self.parser_dict['epochs']=100
        self.parser_dict['lr']=0.01
        self.parser_dict['cuda']=True
        self.parser_dict['log_interval']=3
        self.parser_dict['save_folder']='D:\\python\\BingdianNet\\result'
        self.parser_dict['save_model_name']='my_unet.pth'
        self.parser=Parser(self.parser_dict)#将参数字典赋值给Parser类

        self.batch_size=1
        self.test_batch_size=2
        self.epochs=100
        self.lr=0.01
        self.cuda=True
        self.log_interval=3
        self.save_folder='D:\\python\\BingdianNet\\result\\'
        self.save_model_name='my_unet_test.pth'
        
                
        #训练集做数据增强
        transform_image = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
        ])
        #标签数据处理
        transform_label = transforms.Compose([
        transforms.Grayscale(),
        toBinary(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4938, 0.4933, 0.4880), (0.1707, 0.1704, 0.1672)),
        ])

        #存储图片id，区分训练集与验证集
        ori_img_path = "D:\\python\\BingdianNet\\raw\\unet\\train"
        ori_list = sorted(os.listdir(ori_img_path))
        label_path = "D:\\python\\BingdianNet\\raw\\unet\\label"
        label_list = sorted(os.listdir(label_path))
        train_img_list, val_img_list = ori_list[:24], ori_list[24:]
        train_label_list, val_label_list = label_list[:24], label_list[24:]

        #训练数据读取
        train_dataset = UnetDataset(img_root=ori_img_path,label_root=label_path,img_list=train_img_list,label_list=train_label_list,
        transform=transform_image, target_transform=transform_label)
        #验证数据读取
        val_dataset = UnetDataset(img_root=ori_img_path,label_root=label_path,img_list=val_img_list,label_list=val_label_list,
        transform=transform_image, target_transform=transform_label)
        #加载数据
        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, **kwargs)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.test_batch_size, shuffle=False, **kwargs)
        
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model=UNet(1,1)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()

        self.best_dice = 0.0


    def train(self,epoch):
        self.model.train()
        dice = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            # print("data shape:{}".format(data.dtype))
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target) # This loss is per image
            sig_output = self.sig(output)
            dice += UNetdice.dice_coeff(sig_output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

        dice_acc = 100. * dice / len(self.train_loader)
        print('Train Dice coefficient: {:.2f}%'.format(dice_acc))

    def test(self):
        self.model.eval()
        test_loss = 0
        dice = 0
        iou=0
        global best_dice
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                sig_output = self.sig(output)
                dice += UNetdice.dice_coeff(sig_output, target)
                iou += iou_score(output, target)
                print("iou为",iou)


        test_loss /= len(self.val_loader.dataset)
        dice_acc = 100. * dice / len(self.val_loader)
        iou_acc = 100. * iou / len(self.val_loader)
        print("iou_acc为",iou_acc)
        print('\nTest set: Batch average loss: {:.4f}, Dice Coefficient: {:.2f}%\n'.format(test_loss, dice_acc))

        if dice_acc > self.best_dice:
            torch.save(self.model.state_dict(), self.save_folder + 'UNet\\'+self.save_model_name)
            self.best_dice = dice_acc
            print("======Saving model======")


    def main(self):
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
        ])

        for epoch in range(1, self.epochs + 1):
            
            Segmentation.train(self,epoch)
            Segmentation.test(self)



         
seg_UNet=Segmentation()
seg_UNet.main()

