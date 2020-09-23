#Demo-1-TinyConvWithoutSequential-TinyData.py
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
from load_dataset import custom_tiny

if __name__=='__main__':
    general_results_dir='C:\\Users\\Rachel\\Documents\\Temp\\pytorch-computer-vision\\results'
    
    tot0 = timeit.default_timer()
    run_experiment.DoExperiment(descriptor='TinyConvWithoutSequential_TinyData',
            general_results_dir=general_results_dir,
            custom_net = custom_models_base.TinyConvWithoutSequential,
            custom_net_args = {},
            learning_rate = 1e-3, #default 1e-3
            weight_decay = 1e-7, #default 1e-7
            num_epochs=3, patience = 3,
            batch_size = 1, debug=True,
            use_test_set = False, task = 'train_eval',
            old_params_dir = '',
            chosen_dataset = custom_tiny.TinyData,
            chosen_dataset_args = {})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
