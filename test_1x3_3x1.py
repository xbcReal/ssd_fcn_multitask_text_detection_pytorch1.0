import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        super(Net, self).__init__()  # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1 = nn.Conv2d(2, 2, (3,3),padding=1,dilation=2)
        #self.conv2 = nn.Conv2d(2, 2, (5,3),padding=0)

    # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        x =self.conv1(x)
        #x =self.conv1(x)
        return x


net = Net().double()

input=np.zeros(shape=(1,2,40,40))
input=torch.from_numpy(input).double()
output=net(input)
print(output.size())
