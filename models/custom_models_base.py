#custom_models_base.py
# Copyright (c) 2020 Rachel Lea Ballantyne Draelos
# 
# MIT License
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import numpy as np
import torchvision
import torch, torch.nn as nn
import torchvision.models as models

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from . import components as cts

class TinyConvWithoutSequential(nn.Module):
    """Tiny convolutional neural network.
    This model doesn't use nn.Sequential."""
    def __init__(self):
        super(TinyConvWithoutSequential, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size = (3,3), stride=(3,3), padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = (3,3), stride=(3,3), padding=0)
        self.fc = nn.Linear(in_features=16*3*3,out_features=3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = self.fc(x)
        out = {'out':x}
        return out

class TinyConv(nn.Module):
    """Tiny convolutional neural network.
    Uses nn.Sequential to improve organization."""
    def __init__(self):
        super(TinyConv, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(3, 8, kernel_size = (3,3), stride=(3,3), padding=0),
                            nn.ReLU(inplace=True),
                    
                            nn.Conv2d(8, 16, kernel_size = (3,3), stride=(3,3), padding=0),
                            nn.ReLU(inplace=True))
        self.fc = nn.Linear(in_features=16*3*3,out_features=3)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = self.fc(x)
        out = {'out':x}
        return out

class VGG16(nn.Module):
    """Model with a VGG-16 feature extractor pretrained on ImageNet."""
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*(list(vgg16.children())[:-2]))
        self.conv = nn.Sequential(
                            nn.Conv2d(512, 256, kernel_size = (1,1), stride=(1,1), padding=0),
                            nn.ReLU(inplace=True),
                
                            nn.Conv2d(256, 64, kernel_size = (1,1), stride=(1,1), padding=0),
                            nn.ReLU(inplace=True))
        self.fc = nn.Linear(in_features=64*10*10,out_features=20)
    
    def forward(self, x):
        #for PASCAL VOC, x shape [batch_size, 3, 320, 320]
        x = self.features(x) #for PASCAL VOC, out shape [batch_size, 512, 10, 10]
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = self.fc(x)
        out = {'out':x}
        return out

class VGG16Customizable(nn.Module):
    """Model with a VGG-16 feature extractor.
    <n_outputs>: int, total number of outputs for the model
    <conv_type>: string, defines what kind of conv layer will be used.
        '512to256to64_1x1', '512to64_1x1', or '512to512_3x3'
    <vgginit>: either 'random' or 'pretrained' for randomly-initialized VGG-16
        or VGG-16 pretrained on ImageNet respectively"""
    def __init__(self, n_outputs, conv_type, vgginit):
        super(VGG16Customizable, self).__init__()
        self.conv_type = conv_type
        assert vgginit in ['random','pretrained']
        if vgginit == 'random':
            print('VGG16 feature extractor pretrained=False')
            vgg16 = models.vgg16(pretrained=False)
        elif vgginit == 'pretrained':
            print('VGG16 feature extractor pretrained=True')
            vgg16 = models.vgg16(pretrained=True)
        
        self.features = nn.Sequential(*(list(vgg16.children())[:-2]))
        
        self.conv, flat_out_rep_size = cts.return_conv_layers(conv_type)
        
        self.fc = nn.Linear(in_features=flat_out_rep_size,out_features=n_outputs)
    
    def forward(self, x):
        #for PASCAL VOC, x shape [batch_size, 3, 320, 320]
        x = self.features(x) #for PASCAL VOC, out shape [batch_size, 512, 10, 10]
        x = self.conv(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = self.fc(x)
        out = {'out':x}
        return out
