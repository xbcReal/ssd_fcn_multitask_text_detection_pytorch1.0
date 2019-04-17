import torch
import torch.nn as nn
from ssd.modeling.ssd import SSD
import torch.nn.functional as F



#Inception-style Text Module(ITM)
class ITM(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(ITM,self).__init__()
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.conv2d=nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)
        self.conv3x3=nn.Conv2d(input_channels,256,kernel_size=(3,3),stride=1,padding=1,dilation=1)
        self.relu1=nn.ReLU()
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3x5=nn.Conv2d(input_channels,256,kernel_size=(3,5),stride=1,padding=1,dilation=1)
        self.relu2=nn.ReLU()
        self.bn2 = nn.BatchNorm2d(256)
        self.conv5x3=nn.Conv2d(input_channels,256,kernel_size=(5,3),stride=1,padding=1,dilation=1)
        self.relu3=nn.ReLU()
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3x3_mix=nn.Conv2d(768,self.output_channels,kernel_size=(3,3),stride=1,padding=1)
    def forward(self, x):
        # print('x.size:',x.size())
        #如果feature mapd的尺寸小于5，则不使用ITM模块，直接使用nn.Conv2d模块做正常的3x3卷积
        if(x.size()[-1]<5):
            return self.conv2d(x)
        out_3x3=self.conv3x3(x)
        base_size=out_3x3.size()
        out_3x3=self.bn1(out_3x3)
        out_3x3=self.relu1(out_3x3)

        out_3x5=self.conv3x5(x)
        out_3x5 = self.bn2(out_3x5)
        out_3x5=self.relu2(out_3x5)
        out_3x5=nn.functional.interpolate(out_3x5, size=base_size[2:4],mode='bilinear')

        out_5x3=self.conv5x3(x)
        out_5x3 = self.bn3(out_5x3)
        out_5x3=self.relu2(out_5x3)
        out_5x3=nn.functional.interpolate(out_5x3, size=base_size[2:4],mode='bilinear')
        # print('out_3x3:',out_3x3.size())
        # print('out_3x5:',out_3x5.size())
        # print('out_5x3:',out_5x3.size())
        out_cat=torch.cat([out_3x3,out_3x5,out_5x3],dim=1)
        out_3x3_mix=self.conv3x3_mix(out_cat)
        return out_3x3_mix
class ITM3x5(nn.Module):
    def __init__(self,input_channels,output_channels,regression=True):
        super(ITM3x5,self).__init__()
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.regression = regression
        self.conv2d=nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)
        self.conv3x5=nn.Conv2d(self.input_channels,self.output_channels,kernel_size=(3,5),stride=1,padding=1,dilation=1)
    def forward(self, x):
        # print('x.size:',x.size())
        #如果feature mapd的尺寸小于5，则不使用ITM3x5模块，直接使用nn.Conv2d模块做正常的3x3卷积
        m = self.conv2d(x)
        if(x.size()[-1]<5):
            if not self.regression:
                return m
            else:
                m = F.tanh(m)
            return m
        base_size = m.size()
        out_3x5=self.conv3x5(x)
        out_3x5 = nn.functional.interpolate(out_3x5, size=base_size[2:4], mode='bilinear')
        if not self.regression:
            return out_3x5
        else:
            out_3x5 = F.tanh(out_3x5)
        return out_3x5
class ITM3x3(nn.Module):
    def __init__(self,input_channels,output_channels,regression=True):
        super(ITM3x3,self).__init__()
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.regression = regression
        self.conv2d=nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)
    def forward(self, x):
        # print('x.size:',x.size())
        #如果feature mapd的尺寸小于5，则不使用ITM3x5模块，直接使用nn.Conv2d模块做正常的3x3卷积
        m = self.conv2d(x)

        if not self.regression:
            return m
        else:
            m = 2 * F.softplus(m) - 1

        return m
# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
# 用于添加vggNet的backbone
def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    downsample_layers_index = []
    index = -1
    flag = False
    for v in cfg:
        if v == 'M':
            if flag:
                downsample_layers_index.append(index)
            index += 1
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            downsample_layers_index.append(index)
            index += 1
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            flag = True
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                index += 3
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                index += 2
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    index += 1
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    index += 2
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    index += 2
    downsample_layers_index.append(index)#add conv7
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers,downsample_layers_index


#用于添加prior box的分类和回归层
def add_header(vgg, extra_layers, boxes_per_location, num_classes):
    print('extra_layers',extra_layers)
    regression_headers = []
    classification_headers = []
    vgg_source = [21, -2]
    #vgg[21]是conv_4_3,vgg[-2]是conv7
    for k, v in enumerate(vgg_source):
        # regression_headers += [nn.Conv2d(vgg[v].out_channels,boxes_per_location[k] * 8, kernel_size=3, padding=1,dilation=1)]
        # regression_headers += [ITM3x3(vgg[v].out_channels,boxes_per_location[k] * 8)]
        regression_headers += [ITM(vgg[v].out_channels,boxes_per_location[k] * 8)]
        # classification_headers += [nn.Conv2d(vgg[v].out_channels,boxes_per_location[k] * num_classes, kernel_size=3, padding=1,dilation=1)]
        # classification_headers += [ITM3x3(vgg[v].out_channels,boxes_per_location[k] * num_classes,False)]
        classification_headers += [ITM(vgg[v].out_channels,boxes_per_location[k] * num_classes)]


    #extra_layers是conv_8_2,conv_9_2,conv_10_2,conv_11_2,
    for k, v in enumerate(extra_layers[1::2], 2):
        print('v:',v)
        # regression_headers += [nn.Conv2d(v.out_channels, boxes_per_location[k] * 8, kernel_size=3, padding=1,dilation=1)]
        #regression_headers += [ITM3x3(v.out_channels,boxes_per_location[k] * 8)]
        regression_headers += [ITM(v.out_channels, boxes_per_location[k] * 8)]
        # classification_headers += [nn.Conv2d(v.out_channels, boxes_per_location[k]*num_classes, kernel_size=3, padding=1,dilation=1)]
        # classification_headers += [ITM3x3(v.out_channels,boxes_per_location[k] * num_classes,False)]
        classification_headers += [ITM(v.out_channels, boxes_per_location[k]*num_classes)]
    return regression_headers, classification_headers

#用于添加backbone之后的额外的卷积层
def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        # 'S'用于区别卷积步长是1还是2
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
    # if size == 1024:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers

def build_ssd_model(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES
    size = cfg.INPUT.IMAGE_SIZE
    vgg_base = {
        '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512],
        '1024': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    }
    extras_base = {
        '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
        '1024': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
    }


    boxes_per_location = cfg.MODEL.PRIORS.BOXES_PER_LOCATION

    vgg_config = vgg_base[str(size)]
    extras_config = extras_base[str(size)]
    vgg,downsample_layers_index = add_vgg(vgg_config,batch_norm=False)
    vgg = nn.ModuleList(vgg)

    extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))

    regression_headers, classification_headers = add_header(vgg, extras, boxes_per_location, num_classes=num_classes)
    regression_headers = nn.ModuleList(regression_headers)
    classification_headers = nn.ModuleList(classification_headers)

    return SSD(cfg=cfg,
               vgg=vgg,
               extras=extras,
               classification_headers=classification_headers,
               regression_headers=regression_headers,
               downsample_layers_index=downsample_layers_index)
