import torch
from torchvision import transforms,models
from Net.UNet import UNet
from Utils.Dataset import UnetDataset
from Utils.Transform import toBinary
from Utils.Loss import UNetdice
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
            self.log_interval=parser['log_interval']
            self.save_model_name=parser['save_model_name']
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
        self.parser_dict['log_interval']=3
        self.parser_dict['save_folder']='D:\\python\\BingdianNet\\result'
        self.parser_dict['save_model_name']='my_unet.pth'
        self.parser=Parser(self.parser_dict)#将参数字典赋值给Parser类

        self.batch_size=2
        self.test_batch_size=2
        self.epochs=1
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
        global best_dice
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                sig_output = self.sig(output)
                dice += UNetdice.dice_coeff(sig_output, target)


        test_loss /= len(self.val_loader.dataset)
        dice_acc = 100. * dice / len(self.val_loader)

        print('\nTest set: Batch average loss: {:.4f}, Dice Coefficient: {:.2f}%\n'.format(test_loss, dice_acc))

        if dice_acc > self.best_dice:
            torch.save(self.model.state_dict(), self.save_folder + 'UNet\\'+self.save_model_name)
            self.best_dice = dice_acc
            print("======Saving model======")
    def main(self):
        for epoch in range(1, self.epochs + 1):
            Segmentation.train(self,epoch)
            Segmentation.test(self)



from Net.UNetplus import *
import albumentations as A
#from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from Utils.Dataset import UNetplusDataset
from sklearn.model_selection import train_test_split
from Utils.scripts import AverageMeter,str2bool,count_params
from tqdm import tqdm
from Utils.Eval import iou_score
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.optim import lr_scheduler


class seg_UNetplus(object):
    def __init__(self):     
        self.name='Unetplus'
        self.input_h=96
        self.input_w=96

        self.batch_size=1
        self.test_batch_size=1
        self.epochs=100
        self.min_lr=1e-5
        self.lr=0.001
        self.momentum=0.9
        self.weight_decay=1e-4
        self.nesterov=False

        self.cuda=True
        self.log_interval=3
        self.num_classes=1
        self.deep_supervision=False
        self.input_channels=1
        self.save_folder='D:\\python\\BingdianNet\\result\\unet++\\'
        self.save_model_name='my_unet++.pth'


        self.model = NestedUNet(self.num_classes,self.input_channels,self.deep_supervision)#第一项参数为num_classes，在本数据集中为1
        self.model = self.model.cuda()
        self.params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.criterion= nn.BCEWithLogitsLoss().cuda()
        self.optimizer= optim.SGD(self.params, lr=self.lr, momentum=self.momentum,
                              nesterov=self.nesterov, weight_decay=self.weight_decay)
        self.scheduler=lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.min_lr)
        self.early_stopping=-1


        self.best_iou = 0
        self.trigger = 0

        #数据存储地址
        img_root='D:\\python\\pycharm\\unet\\Unet-pytorch\\raw\\train'
        img_list= sorted(os.listdir(img_root))
        label_root='D:\\python\\pycharm\\unet\\Unet-pytorch\\raw\\train'
        label_list= sorted(os.listdir(label_root))
        train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=41)
        
        #UNet++数据读取器
        #训练数据增强img_rootlabel_list
        # train_transform = Compose([
        #         A.RandomRotate90(),
        #         transforms.Flip(),
        #         OneOf([
        #             transforms.HueSaturationValue(),
        #             transforms.RandomBrightness(),
        #             transforms.RandomContrast(),
        #         ], p=1),
        #         A.Resize(self.input_h, self.input_w),
        #         transforms.Normalize(),
        #     ])

        # val_transform= Compose([
        #     A.Resize(self.input_h, self.input_w),
        #     transforms.Normalize(),
        # ])
        # train_dataset = UNetplusDataset(
        #     img_root=img_root,
        #     label_root=label_root,
        #     img_list=train_img_list,
        #     #label_list=label_list,
        #     transform=train_transform,
        #     target_transform = val_transform
        #     )
        # val_dataset = UNetplusDataset(
        #     img_root=img_root,
        #     label_root=label_root,
        #     img_list=val_img_list,
        #     #label_list=label_list,
        #     transform=train_transform,
        #     target_transform = val_transform
        #     )

        #UNet数据读取器
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
        train_dataset = UnetDataset(img_root=img_root,label_root=label_root,img_list=train_img_list,label_list=train_img_list,
        transform=transform_image, target_transform=transform_label)
        #验证数据读取
        val_dataset = UnetDataset(img_root=img_root,label_root=label_root,img_list=val_img_list,label_list=val_img_list,
        transform=transform_image, target_transform=transform_label)

        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}#num_worker用来加载batch到内存中
        self.train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,**kwargs)
        self.val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,**kwargs)

        #存放模型地址
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        #模型，损失函数，优化器


        
    def train(self):
        avg_meters = {'loss': AverageMeter(),
                    'iou': AverageMeter()}
        self.model.train()
        pbar = tqdm(total=len(self.train_loader))#进度条库
        #for input, target, _ in self.train_loader:#横线是存有image id的字典
        for input, target in self.train_loader:

            input = input.cuda()
            print(input)
            print(input.shape)

            target = target.cuda()

            # compute output
            if self.deep_supervision:
                outputs = self.model(input)
                loss = 0
                for output in outputs:
                    loss += self.criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = self.model(input)
                loss = self.criterion(output, target)
                iou = iou_score(output, target)

            # compute gradient and do optimizing step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])
    def validate(self):
        avg_meters = {'loss': AverageMeter(),
                    'iou': AverageMeter()}

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(total=len(self.val_loader))
            #for input, target, _ in self.val_loader:
            for input, target in self.val_loader:
                input = input.cuda()
                target = target.cuda()

                # compute output
                if self.deep_supervision:
                    outputs = self.model(input)
                    loss = 0
                    for output in outputs:
                        loss += self.criterion(output, target)
                    loss /= len(outputs)
                    iou = iou_score(outputs[-1], target)
                else:
                    output = self.model(input)
                    loss = self.criterion(output, target)
                    iou = iou_score(output, target)

                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])

    def main(self):

        cudnn.benchmark = True

        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_iou', []),
        ])


        for epoch in range(self.epochs):
            print('Epoch [%d/%d]' % (epoch, self.epochs))

            # train for one epoch
            train_log = seg_UNetplus.train(self)
            # evaluate on validation set
            val_log = seg_UNetplus.validate(self)

            if self.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
            elif self.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(val_log['loss'])

            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            log['epoch'].append(self.epochs)
            log['lr'].append(self.lr)
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])

            pd.DataFrame(log).to_csv(self.save_folder + 'log.csv', index=False)

            self.trigger += 1

            if val_log['iou'] > self.best_iou:
                torch.save(self.model.state_dict(), self.save_folder + self.save_model_name)
                self.best_iou = val_log['iou']
                print("=> saved best model")
                self.trigger = 0

            # early stopping
            if self.early_stopping >= 0 and self.trigger >= self.early_stopping:
                print("=> early stopping")
                break

            torch.cuda.empty_cache()

         
seg_UNet=Segmentation()
seg_UNet.main()
# segplus=seg_UNetplus()
# segplus.main()

