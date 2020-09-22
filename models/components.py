#components.py
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
import torch, torch.nn as nn

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def return_conv_layers(conv_type):
    if conv_type == '512to256to64_1x1':
        conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size = (1,1), stride=(1,1), padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 64, kernel_size = (1,1), stride=(1,1), padding=0),
                nn.ReLU(inplace=True))
        flattened_output_dim = 6400 #64*10*10
    
    elif conv_type == '512to64_1x1':
        conv = nn.Sequential(
                nn.Conv2d(512, 64, kernel_size = (1,1), stride=(1,1), padding=0),
                nn.ReLU(inplace=True))
        flattened_output_dim = 6400 #64*10*10
    
    elif conv_type == '512to512_3x3':
        conv = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True))
        flattened_output_dim = 4608 #512*3*3
 
    return conv, flattened_output_dim
