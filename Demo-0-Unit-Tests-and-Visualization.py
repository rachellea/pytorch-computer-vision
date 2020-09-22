#Demo-0-Unit-Tests-and-Visualization.py
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

from src import unit_tests
from load_dataset import custom_pascal

if __name__ == '__main__':
    #Run unit tests
    unit_tests.run_all()
    
    #Run visualization sanity checks
    #Change voc_dataset_dir to a directory in which you want to store the
    #PASCAL VOC 2012 data:
    voc_dataset_dir = 'C:\\Users\\Rachel\\Documents\\Data\\VOC2012'
    #Uncomment the following line if you need to download the VOC 2012 dataset:
    #dataset = torchvision.datasets.VOCSegmentation(voc_dataset_dir, year='2012', image_set='val', download=True)
    
    #The following visualizations demonstrate: (a) the images and ground truth
    #are matched correctly, (b) the mapping from integers to strings (e.g.
    #from 1 to 'airplane') is correct, and (c) the resampling result looks
    #reasonable:
    custom_pascal.visualize_val_set(voc_dataset_dir, resample=False,
                                    images_to_visualize=3)
    custom_pascal.visualize_val_set(voc_dataset_dir, resample=True,
                                    images_to_visualize=3)
    