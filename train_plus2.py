import torch
class seg_UNetplus(object):
    def __init__(self):
        #输入尺寸
        self.input_h=96
        self.input_w=96