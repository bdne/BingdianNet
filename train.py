from Net.UNet import UNet
from Utils.Dataset import UnetDataset

class Parser(object):
        def __init__(self,parser):
            self.batch_size=parser['batch_size']
            self.test_batch_size=parser['test_batch_size']
            self.epochs=parser['epochs']
            self.lr=parser['lr']
            self.no_cuda=parser['no_cuda']
        def get_item(self):
            print(self.epochs) 


class Segmentation(object):
    def __init__(self,data_root):
        self.data_root=data_root
        #实例化参数
        self.parser_dict={}
        self.parser_dict['batch_size']=4
        self.parser_dict['test_batch_size']=2
        self.parser_dict['epochs']=100
        self.parser_dict['lr']=0.01
        self.parser_dict['no_cuda']=True
        self.parser=Parser(self.parser_dict)

    def main(self):
        print(self.parser.epochs)
        dataset=UnetDataset(img_list=['1.tif','2.tif'], label_list=train_label_list,
                            transform=transform_image, target_transform=transform_label)
        train_dataset = UnetDataset(img_list=train_img_list, label_list=train_label_list,
        transform=transform_image, target_transform=transform_label)
        



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
