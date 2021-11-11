import numpy as np
import torch
import torch.nn.functional as F

class UNetdice(object):#Dice系数
    def dice_coeff(target,output_argmax):
        #pred和target是onehot编码，index是标签类别索引矩阵，output_argmax是预测类别索引矩阵
        smooth = 1e-5
        # print("pred:",pred.shape)
        # num = pred.size(0)
        # m1 = pred.view(num, -1)  # Flatten
        # m2 = target.view(num, -1)  # Flatten
        intersection = (target * output_argmax).sum()      #预测分割图与GT分割图点乘，并将元素相乘的结果元素相加求和 
        return (2. * intersection + smooth) / (target.sum() + output_argmax.sum() + smooth)

def iou_score(output, target):#计算交并比
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)