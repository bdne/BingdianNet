import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda")

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction,self).__init__()
        self.cla_loss = Cla_CrossEntropyLoss()
        # self.cla_loss = CrossEntropyLoss_onehot()
    def forward(self,input_cla,target_cla):
        loss = self.cla_loss(input_cla,target_cla)
        return loss
class Cla_CrossEntropyLoss(object):
    def __init__(self,weight=None):
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    def __call__(self,input,target):
        n=input.size()[0]
        target = target.long()
        return torch.sum(self.criterion(input, target)) / n


class CrossEntropyLoss_onehot(object):
    def __init__(self,weight=None):
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    def __call__(self,input,target):
        target = target.cuda().long()
        batch_size, num_classes, width, height = input.size()
        labels = torch.full(size=(batch_size,num_classes,width,height), fill_value=0).cuda()
   
        labels.scatter_(dim=1, index=target, value=1)#one-hot编码,scatter_后面带横线，不然不报错结果不对
        log_prob = F.log_softmax(input, dim=1)#logsoftmax
        loss = -torch.sum(log_prob * labels) / (batch_size*width*height)#交叉熵 per image

        return loss