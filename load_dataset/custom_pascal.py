#custom_pascal.py
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torchvision
from torch.utils.data import Dataset, DataLoader

from . import utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class PascalVOC2012(Dataset):
    def __init__(self, setname, voc_dataset_dir, sbd_dataset_dir):
        """Wrapper for the Pascal VOC 2012 dataset class provided in torchvision.
        Returns the data examples as a dict with key 'data' for the image,
        'seglabel' for the segmentation ground truth, and 'label' for
        the classification label vector.
        Variables:
       <setname> can be: 'train' (Pascal VOC 2012 train + SBD train)
                         'val' (Pascal VOC 2012 val)"""
        self.setname = setname
        assert setname in ['train','val']
        self.label_meanings = return_label_names()
        
        #Define dataset
        if setname == 'train':
            #In the training set, combine PASCAL VOC 2012 with SBD
            self.dataset = [torchvision.datasets.VOCSegmentation(voc_dataset_dir, year='2012',image_set='train',download=False),
                            #SBD image set train_noval excludes VOC 2012 val images
                           torchvision.datasets.SBDataset(sbd_dataset_dir, image_set='train_noval', mode='segmentation',download=False)]
        elif setname == 'val':
            self.dataset = [torchvision.datasets.VOCSegmentation(voc_dataset_dir, year='2012',image_set='val',download=False)]
    
    def __len__(self):
        if self.setname == 'train':
            return len(self.dataset[0])+len(self.dataset[1])
        elif self.setname == 'val':
            return len(self.dataset[0])
    
    def __getitem__(self, idx):
        """Return a single example at index <idx>.
        The example is a dict with keys 'data' (value: Tensor of the image),
        'seglabel' (value: np array of segmentation gr truth),
        'label' (value: Tensor multi-hot vector of gr truth class labels).
        
        Note that by default, the torchvision data loader returns
        (image, target) each as a PIL image, where target is the image
        segmentation ground truth."""
        if idx < len(self.dataset[0]):
            chosen_dataset = self.dataset[0]
        else:
            chosen_dataset = self.dataset[1]
            idx = idx - len(self.dataset[0])
        #Example:
        #>>> imagepil
        #<PIL.Image.Image image mode=RGB size=334x500 at 0x22CCA12E080>
        #>>> targetpil
        #<PIL.PngImagePlugin.PngImageFile image mode=P size=334x500 at 0x22CCFB35BA8>
        imagepil, targetpil = chosen_dataset[idx]
        
        #resample the ground truth segmentation to 320 x 320
        #Why? Because all the images in this dataset are slightly different shapes
        #but to make a batch, they all need the same height and width.
        #Alternative to resampling could be random cropping to 320 x 320.
        targetpil = targetpil.resize((320,320), resample=PIL.Image.NEAREST)
        target = np.array(targetpil).astype('int').squeeze()
        label = torch.Tensor(get_label_vector(targetpil))
        
        #convert image to Tensor and normalize
        image = utils.to_tensor_and_normalize(imagepil) #e.g. out shape torch.Size([3, 500, 334])
        #resample the image to 320 x 320
        image = image.unsqueeze(0)
        image = torch.nn.functional.interpolate(image,size=(320,320),mode='bicubic')
        image = image.squeeze()
        
        sample = {'data':image, #preprocessed image, for input into NN
                  'seglabel':target,
                  'label':label,
                  'img_idx':idx}
        return sample

######################
# Data Preprocessing #----------------------------------------------------------
######################
def get_label_vector(targetpil): #Done with testing
    """From <targetpil> which is a segmentation ground truth, return a
    Python list of 0 or 1 ints (multi-hot vector) representing the classification
    ground truth"""
    classes = set(np.array(targetpil).flatten().tolist())
    label = [0]*20
    #>>> [x for x in range(1,21)]
    #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for classnum in range(1,21):
        if classnum in classes:
            #Python is zero-indexed so we have to subtract one
            label[classnum-1] = 1
    return label

def return_label_names():
    """Return the class names in order as a Python list.
      The classes are in alphabetical order:
        1 = airplane          2 = bicycle
        3 = bird              4 = boat
        5 = bottle            6 = bus
        7 = car               8 = cat
        9  = chair            10 = cow
        11 = dining table     12 = dog
        13 = horse            14 = motorbike
        15 = person           16 = potted plant
        17 = sheep            18 = sofa
        19 = train            20 = tv/monitor"""
    label_names = ['airplane','bicycle','bird','boat','bottle','bus','car','cat',
            'chair','cow','dining_table','dog','horse','motorbike','person',
            'potted_plant','sheep','sofa','train','tv_monitor']
    assert label_names == sorted(label_names)
    return label_names

def get_all_labels_string(labels_list): #Done with testing
    """Return a string containing the alphabetized labels present in the
    <labels_list>. The <labels_list> is a list of ints of length 20
    indicating which labels are present.
    E.g. if the array contains a 1 at position 0, then return 'airplane.'
    Used in making plots for sanity checks."""
    label_names = return_label_names()
    labels_present = []
    for list_idx in range(0,20):
        if labels_list[list_idx] == 1:
            labels_present.append(label_names[list_idx])
    return ' '.join(sorted(labels_present))
    
#################
# Visualization #---------------------------------------------------------------
#################
def visualize_val_set(voc_dataset_dir, resample, images_to_visualize):
    """Make plots of the images and the ground truth segmentation from the
    validation set using matplotlib's interactive viewer.
    
    <voc_dataset_dir>: path to the directory containing the VOC 2012 dataset
    <resample>: if True, resample to (320,320) before doing the visualization
        since currently you are doing resampling as a preprocessing step.
    <images_to_visualize>: int for the total number of images to visualize"""
    #When we load with no transforms, both the image and target will be
    #PIL images with pixel values in 0 - 255
    dataset = torchvision.datasets.VOCSegmentation(voc_dataset_dir, year='2012',
                                            image_set='val', download=False)
    for img in range(0,images_to_visualize):
        imagepil, targetpil = dataset[img]
        if resample:
            imagepil = imagepil.resize((320,320), resample=PIL.Image.BICUBIC)
            targetpil = targetpil.resize((320,320), resample=PIL.Image.NEAREST)
        #Plot
        visualize_target_and_image(imagepil,targetpil)
    
def visualize_target_and_image(imagepil, targetpil):
    """Display segmentation class labels as a color image next to the original
    image"""
    image = np.array(imagepil)
    target = np.array(targetpil)
    labels_list = get_label_vector(targetpil)
    label_string = get_all_labels_string(labels_list)
    fig, ax = plt.subplots(nrows=1,ncols=2)
    ax[0].imshow(image/255.0)
    ax[1].imshow(visualize_target(target))
    plt.title(label_string)
    plt.show()

def visualize_target(target, plot=False):
    #Modified from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
    """Transform segmentation class labels into a color image and return image.
    Args:
        target (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = target.copy()
    g = target.copy()
    b = target.copy()
    n_classes = 21
    for ll in range(0, n_classes):
        r[target == ll] = label_colours[ll, 0]
        g[target == ll] = label_colours[ll, 1]
        b[target == ll] = label_colours[ll, 2]
    rgb = np.zeros((target.shape[0], target.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    return rgb

def get_pascal_labels():
    #From https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )
