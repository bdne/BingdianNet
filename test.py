import torch
import cv2
from torchvision import transforms
from Net.UNet import UNet
from Utils.Dataset import UnetTestDataset
import os
import torch.nn as nn
class Segmentation(object):
    def __init__(self):
        self.test_batch_size=1
        self.cuda=True
        self.save_folder='D:\\python\\BingdianNet\\result\\'
        self.save_model_name='my_unet.pth'
        
        transform_image = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
        ])


        ori_img_path = "D:\\python\\BingdianNet\\raw\\test"
        test_img_list = sorted(os.listdir(ori_img_path))


        #测试数据读取
        test_dataset = UnetTestDataset(img_root=ori_img_path,img_list=test_img_list, transform=transform_image)
        #加载数据
        kwargs = {'num_workers': 0, 'pin_memory': False} if self.cuda else {}
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.test_batch_size, shuffle=False, **kwargs)


        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.model=UNet(1,1)
        self.model.load_state_dict(torch.load('.\\result\\my_unet.pth'))

        self.model = self.model.to(self.device)
        self.pic=[]
        self.m=nn.Sigmoid()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, name) in enumerate(self.test_loader):
                data = data.to(self.device)
                output = self.model(data)
                output = self.m(output)
                # print(output.shape)
                self.pic.append((output, name))
        return self.pic



    def main(self):
        Segmentation.test(self)
        i=0
        for img_tensor in self.pic:
            img = img_tensor[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = img * 255
            # print(img.shape)
            # img = Image.fromarray(img)
            cv2.imwrite(".\\output\\UNet\\%s.jpg" % (img_tensor[1]), img)
            i = i+1

seg_UNet=Segmentation()
seg_UNet.main()
