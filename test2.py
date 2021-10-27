

import torch
from torchvision import transforms,models
import os
import torch.optim as optim
import torch.nn as nn
from Net.UNetplus import *
from Utils.Dataset import UNetplusDataset
from sklearn.model_selection import train_test_split
from Utils.scripts import AverageMeter,str2bool,count_params
from tqdm import tqdm
from Utils.Eval import iou_score
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.optim import lr_scheduler
from Utils.Transform import toBinary
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
import cv2



class toBinary(object):
    def __call__(self, label):
        label = np.array(label)
        # print(image)
        label = label * (label > 127)
        label = Image.fromarray(label)
        return label

transform_image = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
])

transform_label = transforms.Compose([
    transforms.Grayscale(),
    toBinary(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4938, 0.4933, 0.4880), (0.1707, 0.1704, 0.1672)),
])

device = torch.device("cuda")
m = nn.Sigmoid()
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, name) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            output = m(output)
            # print(output.shape)
            pic.append((output, name))
    return pic



pic=[]
def main():

    num_classes=1
    deep_supervision=False
    input_channels=1
    save_folder='D:\\python\\BingdianNet\\result\\unet++\\'
    save_model_name='my_unet++.pth'



    # create model

    model = NestedUNet(num_classes,input_channels,deep_supervision)#第一项参数为num_classes，在本数据集中为1
    model = model.cuda()
    cudnn.benchmark = True



    # Data loading code
    ori_img_path = "D:\\python\\BingdianNet\\raw\\unet\\test"
    test_img_list = sorted(os.listdir(ori_img_path))

    class UnetTestDataset(Dataset):
        """
        You need to inherit nn.Module and
        overwrite __getitem__ and __len__ methods.
        """

        def __init__(self, img_root=ori_img_path,
                     img_list=None, transform=None):
            assert img_root is not None, 'Must specify img_root and label_root!'

            self.img_root = img_root
            self.img_list = img_list
            self.transform = transform

        def __getitem__(self, index):
            image = Image.open(os.path.join(self.img_root, self.img_list[index]))
            image_name = self.img_list[index][:-4]
            if self.transform is not None:
                image = self.transform(image)

            return image, image_name

        def __len__(self):
            return len(self.img_list)

    model.load_state_dict(torch.load(save_folder+save_model_name))
    model.eval()
    transform_image = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
    ])

    test_dataset = UnetTestDataset(img_list=test_img_list, transform=transform_image)


    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('inputs', config['dataset'], 'images'),
    #     mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # batch_size=config['batch_size'],
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False)

    # avg_meter = AverageMeter()
    #
    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    # with torch.no_grad():
    #     for input, target in tqdm(test_loader, total=len(test_loader)):
    #         input = input.cuda()
    #         target = target.cuda()
    #
    #         # compute output
    #         if config['deep_supervision']:
    #             output = model(input)[-1]
    #         else:
    #             output = model(input)
    #
    #         iou = iou_score(output, target)
    #         avg_meter.update(iou, input.size(0))
    #
    #         output = torch.sigmoid(output).cpu().numpy()
    #
    #         for i in range(len(output)):
    #             for c in range(config['num_classes']):
    #                 cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
    #                             (output[i, c] * 255).astype('uint8'))
    #
    # print('IoU: %.4f' % avg_meter.avg)
    #
    # torch.cuda.empty_cache()
    test(model, device, test_loader)
    i=0
    for img_tensor in pic:
        print(img_tensor[1])
        print(img_tensor[0].shape)
        # print(img_tensor[0].squeeze(0).shape)
        img = img_tensor[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * 255
        # print(img.shape)
        # img = Image.fromarray(img)

   
        cv2.imwrite(".\\output\\UNet++\\%s.jpg"% (img_tensor[1]), img)
        i = i+1
        print("over")


if __name__ == '__main__':
    main()
