#unit_tests.py
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

import os
import PIL
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import sys
sys.path.append('..')
from load_dataset import custom_pascal
from load_dataset import utils

#########
# Tests #-----------------------------------------------------------------------
#########
def test_to_tensor_and_normalize():
    red = np.array([[255,16],  #0,1
                    [23,9]])   #2,3
    green = np.array([[133,230],
                      [0,87]])
    blue = np.array([[200,67],
                     [189,255]])
    x = np.stack([red,green,blue],axis=2) #shape (2, 2, 3)
    #x printed out:
    #>>> x
    # array([[[255, 133, 200],
    #         [ 16, 230,  67]],
    #        [[ 23,   0, 189],
    #         [  9,  87, 255]]])
    #i.e.
    # array([[[red0, green0, blue0],
    #         [red1, green1, blue1]],
    #        [[red2, green2, blue2],
    #         [red3, green3, blue3]]])
    
    #Documentation from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#ToTensor
    #class ToTensor() converts a PIL Image (H x W x C) in the range
    #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    
    #To calculate the correct answer manually, first you go from [0,255]
    #to [0,1] which is the process of converting from PIL to Tensor.
    #Then you normalize with the ImageNet mean=[0.485, 0.456, 0.406]
    #and  std=[0.229, 0.224, 0.225]
    #Here is x divided by 255 to get to range [0,1]:
    #>>> x/255
    # array([[[1.        ,   0.52156863,   0.78431373],
    #         [0.0627451 ,   0.90196078,   0.2627451 ]],
    #        [[0.09019608,   0.        ,   0.74117647],
    #         [0.03529412,   0.34117647,   1.        ]]])
    
    #Normalization: subtract mean and divide by stdev
    # array([[[(1.-0.485)/0.229        ,   (0.52156863-0.456)/0.224,   (0.78431373-0.406)/0.225],
    #         [(0.0627451-0.485)/0.229 ,   (0.90196078-0.456)/0.224,   (0.2627451-0.406)/0.225 ]],
    #        [[(0.09019608-0.485)/0.229,   (0.-0.456)/0.224        ,   (0.74117647-0.406)/0.225],
    #         [(0.03529412-0.485)/0.229,   (0.34117647-0.456)/0.224,   (1.-0.406)/0.225        ]]])
    correct = torch.Tensor([[[2.2489082969432315,0.29271709821428554,1.6813943555555555],
                            [-1.8439078602620087,1.9908963392857142,-0.6366884444444445]],
                           [[-1.7240345851528383,-2.0357142857142856,1.4896731999999997],
                            [-1.963781135371179,-0.5126050446428572, 2.6399999999999997]]])
    #Now because the Image goes from (H x W x C) to (C x H x W) we
    #need to transpose the correct answer.
    correct = correct.transpose(0,2).transpose(1,2)
    
    #Calculate output and compare to correct
    xpil = PIL.Image.fromarray(x.astype('uint8'))
    output = utils.to_tensor_and_normalize(xpil)
    assert torch.allclose(output,correct)
    print('Passed test_to_tensor_and_normalize()')

def test_get_label_vector():
    #Test A
    fake_targetA = np.array([[0,1,3],[16,20,0]]).astype('uint8')
    fake_targetpilA = PIL.Image.fromarray(fake_targetA)
    outputA = custom_pascal.get_label_vector(fake_targetpilA)
    #           1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    correctA = [1,0,1,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert outputA==correctA
    
    #Test B
    fake_targetB = np.array([[0,0,0],[0,0,0]]).astype('uint8')
    fake_targetpilB = PIL.Image.fromarray(fake_targetB)
    outputB = custom_pascal.get_label_vector(fake_targetpilB)
    #           1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    correctB = [0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert outputB==correctB
    
    #Test C
    fake_targetC = np.array([[2,12,13],[19,2,2]]).astype('uint8')
    fake_targetpilC = PIL.Image.fromarray(fake_targetC)
    outputC = custom_pascal.get_label_vector(fake_targetpilC)
    #           1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    correctC = [0,1,0,0,0,0,0,0,0,0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    assert outputC==correctC
    print('Passed test_get_label_vector()')

def test_get_all_labels_string():
    assert custom_pascal.get_all_labels_string([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])=='airplane'
    assert custom_pascal.get_all_labels_string([0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])=='bird boat sofa'
    print('Passed test_get_all_labels_string()')

########
# Meta #------------------------------------------------------------------------
########
def run_all():   
    test_to_tensor_and_normalize()
    test_get_label_vector()
    test_get_all_labels_string()