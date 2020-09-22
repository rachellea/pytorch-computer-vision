#Demo-4-VGG16Customizable-PASCAL.py
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

import timeit

from src import run_experiment
from models import custom_models_base
from load_dataset import custom_pascal

if __name__=='__main__':
    for conv_type in ['512to256to64_1x1','512to64_1x1','512to512_3x3']:
        for vgginit in ['random','pretrained']:
            tot0 = timeit.default_timer()
            run_experiment.DoExperiment(descriptor='VGG16Customizable_'+conv_type+'_'+vgginit+'_PASCAL',
                    general_results_dir='C:\\Users\\Rachel\\Documents\\CarinLabResults\\pytorch-computer-vision\\results',
                    custom_net = custom_models_base.VGG16Customizable,
                    custom_net_args = {'n_outputs':20,'conv_type':conv_type,'vgginit':vgginit},
                    learning_rate = 1e-3, #default 1e-3
                    weight_decay = 1e-7, #default 1e-7
                    num_epochs=100, patience = 15,
                    batch_size = 64, debug=False,
                    use_test_set = False, task = 'train_eval',
                    old_params_dir = '',
                    chosen_dataset = custom_pascal.PascalVOC2012,
                    chosen_dataset_args = {'voc_dataset_dir':'C:\\Users\\Rachel\\Documents\\Data\\VOC2012',
                                       'sbd_dataset_dir':'C:\\Users\\Rachel\\Documents\\Data\\SBD'})
            tot1 = timeit.default_timer()
            print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
