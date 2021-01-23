#run_experiment.py
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
import timeit
import datetime
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import torch, torch.nn as nn

from . import evaluate

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

import warnings
warnings.filterwarnings('ignore')

class DoExperiment(object):
    def __init__(self, descriptor, general_results_dir,
                 custom_net, custom_net_args,
                 learning_rate, weight_decay,
                 num_epochs, patience, batch_size, debug, use_test_set, task,
                 old_params_dir, chosen_dataset, chosen_dataset_args):
        """Variables:
        <descriptor>: string describing the experiment. This descriptor will
            become part of a directory name, so it's a good idea not to
            include spaces in this string.
        <general_results_dir>: string. path to directory in which to store
            results.
        <custom_net>: class defining a model. This class must inherit from
            nn.Module.
        <custom_net_args>: dictionary where keys correspond to custom net
            input arguments, and values are the desired values.
        <num_epochs>: int for the maximum number of epochs to train.
        <patience>: int for the number of epochs for which the validation set
            loss must fail to decrease in order to cause early stopping.
        <batch_size>: int for number of examples per batch
        <debug>: if True, use 0 num_workers so that you can run the script
            within the Python debugger on Windows in Anaconda. (If you try
            to do multiprocessing in an interactive environment on Windows
            you get a spec error.)
        <use_test_set>: if True, then run model on the test set. If False, use
            only the training and validation sets. This is meant as an extra
            precaution against accidentally running anything on the test set.
        <task>:
            'train_eval': train and evaluate a new model.
                If <use_test_set> is False, then this will train and evaluate
                a model using only the training set and validation set,
                respectively.
                If <use_test_set> is True, then additionally the test set
                performance will be calculated for the best validation epoch.
            'restart_train_eval': restart training and evaluation of a model
                that wasn't done training (e.g. a model that died accidentally)
            'predict_on_valid': load a trained model and make predictions on
                the validation set using that model.
            'predict_on_test': load a trained model and make predictions on
                the test set using that model.
        <old_params_dir>: this is only needed if <task> is 'restart_train_eval',
            'predict_on_valid', or 'predict_on_test.' This is the path to the
            parameters that will be loaded in to the model.
        <chosen_dataset>: Dataset class that inherits from
            torch.utils.data.Dataset.
        <chosen_dataset_args>: dict of args to be passed to the <chosen_dataset>
            class"""
        self.descriptor = 'Model-'+descriptor
        print(self.descriptor)
        self.general_results_dir = general_results_dir
        self.set_up_results_dirs() #Results dirs for output files and saved models
        self.custom_net = custom_net
        self.custom_net_args = custom_net_args
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_model_every_epoch=False
        
        #num_workers is number of threads to use for data loading
        if debug:
            self.num_workers = 0
            self.batch_size = 1
        else:
            self.num_workers = 16
            self.batch_size = batch_size
        print('num_workers =',self.num_workers)
        print('batch_size =',self.batch_size)
        
        #Set Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device =',str(self.device))
        
        #Set Task
        self.use_test_set = use_test_set
        self.task = task
        assert self.task in ['train_eval','restart_train_eval','predict_on_valid','predict_on_test']
        if self.task in ['restart_train_eval','predict_on_valid','predict_on_test']:
            self.old_params_dir = old_params_dir
        
        #Data and Labels
        if self.task in ['train_eval','restart_train_eval','predict_on_valid']:
            self.dataset_train = chosen_dataset('train',**chosen_dataset_args)
            self.dataset_valid = chosen_dataset('val',**chosen_dataset_args)
            #Get label meanings, a list of descriptive strings for the labels
            #in order:
            self.label_meanings = self.dataset_train.label_meanings
        if self.use_test_set:
            assert False, 'Test set only available via a server for VOC2012'
        
        #Tracking losses and evaluation results
        if self.task in ['train_eval','predict_on_valid']:
            self.train_loss = np.zeros((self.num_epochs))
            self.valid_loss = np.zeros((self.num_epochs))
            self.eval_results_valid, self.eval_results_test = evaluate.initialize_evaluation_dfs(self.label_meanings, self.num_epochs)
        elif self.task == 'restart_train_eval':
            base_old_results_path = os.path.split(os.path.split(old_params_dir)[0])[0]
            self.train_loss = np.load(os.path.join(base_old_results_path, 'train_loss.npy'))
            self.valid_loss = np.load(os.path.join(base_old_results_path, 'valid_loss.npy'))
            self.eval_results_valid, self.eval_results_test = evaluate.load_existing_evaluation_dfs(self.label_meanings, self.num_epochs, base_old_results_path, self.descriptor)
        
        #For early stopping
        self.initial_patience = patience
        self.patience_remaining = patience
        self.best_valid_epoch = 0
        self.min_val_loss = np.inf
        
        #Run everything
        self.run_model()
    
    ### Methods ###
    def set_up_results_dirs(self):
        self.date_and_descriptor = datetime.datetime.today().strftime('%Y-%m-%d')+'_'+self.descriptor
        self.results_dir = os.path.join(self.general_results_dir,self.date_and_descriptor)
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        self.params_dir = os.path.join(self.results_dir,'params')
        if not os.path.isdir(self.params_dir):
            os.mkdir(self.params_dir)
        self.backup_dir = os.path.join(self.results_dir,'backup')
        if not os.path.isdir(self.backup_dir):
            os.mkdir(self.backup_dir)
        
    def run_model(self):
        self.model = self.custom_net(**self.custom_net_args).to(self.device)
        
        #optimizer: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
        momentum = 0.99
        print('Running with optimizer lr='+str(self.learning_rate)+', momentum='+str(round(momentum,2))+' and weight_decay='+str(self.weight_decay))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        
        if self.task in ['restart_train_eval','predict_on_valid']:
            print('For',self.task,'loading model params and optimizer state from ',self.old_params_dir)
            check_point = torch.load(self.old_params_dir)
            self.model.load_state_dict(check_point['params'])
            self.optimizer.load_state_dict(check_point['optimizer'])
        
        if self.task in ['restart_train_eval','predict_on_valid','predict_on_test']:
            #e.g. if you load epoch 51, then start up again on epoch 52
            start_epoch = int(self.old_params_dir.split('epoch')[1])+1
        else:
            start_epoch = 0
        
        if self.task in ['train_eval', 'restart_train_eval']:
            train_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)
            valid_dataloader = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
            for epoch in range(start_epoch, self.num_epochs):  # loop over the dataset multiple times
                print('Epoch',epoch)
                t0 = timeit.default_timer()
                self.train(train_dataloader, epoch)
                self.valid(valid_dataloader, epoch)
                if self.save_model_every_epoch: self.save_model(epoch)
                self.save_evals(epoch)
                if self.patience_remaining <= 0:
                    print('No more patience (',self.initial_patience,') left at epoch',epoch)
                    print('--> Implementing early stopping. Best epoch was:',self.best_valid_epoch)
                    break
                t1 = timeit.default_timer()
                print('Epoch',epoch,'time:',round((t1 - t0)/60.0,2),'minutes')
        if self.task=='predict_on_valid': self.valid(DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers), epoch=0)
        if self.use_test_set: self.test(DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers))
        self.save_final_summary()
    
    def train(self, dataloader, epoch):
        self.model.train()
        epoch_loss, pred_epoch, gr_truth_epoch = self.iterate_through_batches(self.model, dataloader, epoch, training=True)
        self.train_loss[epoch] = epoch_loss
        self.plot_roc_and_pr_curves('train', epoch, pred_epoch, gr_truth_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Train Loss', epoch_loss))
        
    def valid(self, dataloader, epoch):
        self.model.eval()
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch = self.iterate_through_batches(self.model, dataloader, epoch, training=False)
        self.valid_loss[epoch] = epoch_loss
        self.eval_results_valid = evaluate.evaluate_all(self.eval_results_valid, epoch,
            self.label_meanings, gr_truth_epoch, pred_epoch)
        self.early_stopping_check(epoch, pred_epoch, gr_truth_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Valid Loss', epoch_loss))
    
    def early_stopping_check(self, epoch, val_pred_epoch, val_gr_truth_epoch):
        """Check whether criteria for early stopping are met and update
        counters accordingly"""
        val_loss = self.valid_loss[epoch]
        if (val_loss < self.min_val_loss) or epoch==0: #then save parameters
            self.min_val_loss = val_loss
            if not self.save_model_every_epoch: self.save_model(epoch) 
            self.best_valid_epoch = epoch
            self.patience_remaining = self.initial_patience
            print('model saved, val loss',val_loss)
            self.plot_roc_and_pr_curves('valid', epoch, val_pred_epoch, val_gr_truth_epoch)
            self.save_all_pred_probs('valid', epoch, val_pred_epoch, val_gr_truth_epoch)
        else:
            self.patience_remaining -= 1
    
    def save_model(self, epoch):
        check_point = {'params': self.model.state_dict(),                            
                           'optimizer': self.optimizer.state_dict()}
        torch.save(check_point, os.path.join(self.params_dir, self.descriptor+'_epoch'+str(epoch)))        
    
    def test(self, dataloader):
        epoch = self.best_valid_epoch
        model = self.custom_net(**self.custom_net_args).to(self.device).eval()
        params_path = os.path.join(self.old_params_dir,self.descriptor)
        print('For test set predictions, loading model params from params_path=',params_path)
        check_point = torch.load(params_path)
        model.load_state_dict(check_point['params'])
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch = self.iterate_through_batches(model, dataloader, epoch, training=False)
        self.eval_results_test = evaluate.evaluate_all(self.eval_results_test, epoch,
            self.label_meanings, gr_truth_epoch, pred_epoch)
        self.plot_roc_and_pr_curves('test', epoch, pred_epoch, gr_truth_epoch)
        self.save_all_pred_probs('test', epoch, pred_epoch, gr_truth_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Test Loss', epoch_loss))
    
    def iterate_through_batches(self, model, dataloader, epoch, training):
        epoch_loss = 0
        
        #Initialize numpy arrays for storing results. examples x labels
        #Do NOT use concatenation, or else you will have memory fragmentation.
        num_examples = len(dataloader.dataset)
        num_labels = len(self.label_meanings)
        pred_epoch = np.zeros([num_examples,num_labels])
        gr_truth_epoch = np.zeros([num_examples,num_labels])        
        
        for batch_idx, batch in enumerate(dataloader):
            data = batch['data'].to(self.device)
            gr_truth = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            if training:
                out_dict = model(data)
            else:
                with torch.set_grad_enabled(False):
                   out_dict = model(data)
            
            loss_func = nn.BCEWithLogitsLoss() #includes application of sigmoid for numerical stability
            loss = loss_func(out_dict['out'], gr_truth)
            
            if training:
                loss.backward()
                self.optimizer.step()   
            
            epoch_loss += loss.item()
            torch.cuda.empty_cache()
            
            #Save predictions and ground truth across batches
            #Note that torch.nn.Sigmoid(somedata) doesn't work. You first
            #define sigmoid = torch.nn.Sigmoid() and then you can call
            #sigmoid(somedata)
            sigmoid = torch.nn.Sigmoid()
            pred = sigmoid(out_dict['out'].data).detach().cpu().numpy()
            gr_truth = gr_truth.detach().cpu().numpy()
            
            start_row = batch_idx*self.batch_size
            stop_row = min(start_row + self.batch_size, num_examples)
            pred_epoch[start_row:stop_row,:] = pred #pred_epoch is e.g. [25355,80] and pred is e.g. [1,80] for a batch size of 1
            gr_truth_epoch[start_row:stop_row,:] = gr_truth #gr_truth_epoch has same shape as pred_epoch
            
            #the following line to empty the cache is helpful in order to
            #reduce memory usage and avoid OOM error:
            torch.cuda.empty_cache()
        
        #Return loss and classification predictions and classification gr truth
        return epoch_loss, pred_epoch, gr_truth_epoch
    
    def plot_roc_and_pr_curves(self, setname, epoch, pred_epoch, gr_truth_epoch):
        outdir = os.path.join(self.results_dir,'curves')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        evaluate.plot_roc_curve_multi_class(label_meanings=self.label_meanings,
                    y_test=gr_truth_epoch, y_score=pred_epoch,
                    outdir = outdir, setname = setname, epoch = epoch)
        evaluate.plot_pr_curve_multi_class(label_meanings=self.label_meanings,
                    y_test=gr_truth_epoch, y_score=pred_epoch,
                    outdir = outdir, setname = setname, epoch = epoch)
    
    def save_all_pred_probs(self, setname, epoch, pred_epoch, gr_truth_epoch):
        outdir = os.path.join(self.results_dir,'pred_probs')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        (pd.DataFrame(pred_epoch,columns=self.label_meanings)).to_csv(os.path.join(outdir, setname+'_predprob_ep'+str(epoch)+'.csv'))
        (pd.DataFrame(gr_truth_epoch,columns=self.label_meanings)).to_csv(os.path.join(outdir, setname+'_grtruth_ep'+str(epoch)+'.csv'))
        
    def save_evals(self, epoch):
        evaluate.save(self.eval_results_valid, self.results_dir, self.descriptor+'_valid')
        if self.use_test_set: evaluate.save(self.eval_results_test, self.results_dir, self.descriptor+'_test')
        evaluate.plot_learning_curves(self.train_loss, self.valid_loss, self.results_dir, self.descriptor)
                            
    def save_final_summary(self):
        evaluate.save_final_summary(self.eval_results_valid, self.best_valid_epoch, 'valid', self.general_results_dir, self.date_and_descriptor)
        if self.use_test_set: evaluate.save_final_summary(self.eval_results_test, self.best_valid_epoch, 'test', self.general_results_dir, self.date_and_descriptor)
        evaluate.clean_up_output_files(self.best_valid_epoch, self.results_dir)
