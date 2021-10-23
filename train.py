import torch
from Net.UNet import UNet
from Utils.Dataset import UnetDataset
from torchvision import transforms,models
from Utils.Transform import toBinary
import os
import torch.optim as optim
import torch.nn as nn
class Parser(object):
        def __init__(self,parser):
            self.batch_size=parser['batch_size']
            self.test_batch_size=parser['test_batch_size']
            self.epochs=parser['epochs']
            self.lr=parser['lr']
            self.cuda=parser['cuda']
            self.save_folder=parser['save_folder']
        def get_item(self):
            print(self.epochs) 


class Segmentation(object):
    def __init__(self):

        #实例化参数
        self.parser_dict={}#定义一个参数字典
        self.parser_dict['batch_size']=4
        self.parser_dict['test_batch_size']=2
        self.parser_dict['epochs']=100
        self.parser_dict['lr']=0.01
        self.parser_dict['cuda']=True
        self.parser_dict['save_folder']='D:\\python\\BingdianNet\\result'
        self.parser=Parser(self.parser_dict)#将参数字典赋值给Parser类

                
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
        ori_img_path = "D:\\python\\BingdianNet\\raw\\train"
        ori_list = sorted(os.listdir(ori_img_path))
        label_path = "D:\\python\\BingdianNet\\raw\\label"
        label_list = sorted(os.listdir(label_path))
        train_img_list, val_img_list = ori_list[:24], ori_list[24:]
        train_label_list, val_label_list = label_list[:24], label_list[24:]

        #训练数据读取
        train_dataset = UnetDataset(img_list=train_img_list, label_list=train_label_list,
        transform=transform_image, target_transform=transform_label)
        #验证数据读取
        val_dataset = UnetDataset(img_list=val_img_list, label_list=val_label_list,
        transform=transform_image, target_transform=transform_label)
        #加载数据
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.test_batch_size, shuffle=False, **kwargs)
        
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model=UNet(1,1)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()
        best_dice = 0.0


    def train(self):
        self.model.train()
        dice = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            # print("data shape:{}".format(data.dtype))
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target) # This loss is per image
            sig_output = self.sig(output)
            dice += dice_coeff(sig_output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    dice_acc = 100. * dice / len(train_loader)
    print('Train Dice coefficient: {:.2f}%'.format(dice_acc))

    def main(self):

        



seg=Segmentation('/home')
seg.main()

# #parser类实例化
# parser_dict={}
# parser_dict['batch_size']=4
# parser_dict['test_batch_size']=2
# parser_dict['epochs']=100
# parser_dict['lr']=0.01
# parser_dict['no_cuda']=True
# parser=Parser(parser_dict)
# parser.get_item()
# print(parser.lr)
