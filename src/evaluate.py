#evaluate.py
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

#Imports
import os
import copy
import time
import shutil
import numpy as np
import pandas as pd
import sklearn.metrics

import matplotlib
import matplotlib.pyplot as plt

#######################
# Reporting Functions #---------------------------------------------------------
#######################
def initialize_evaluation_dfs(all_labels, num_epochs):
    """Create empty eval_dfs_dict. The keys are 'accuracy', 'auroc', and
    'avg_precision' and the values are pandas dataframes that store the
    performance metric across epochs.
    Variables
    <all_labels>: a list of strings describing the labels in order
    <num_epochs>: int for total number of epochs"""
    if len(all_labels)==2:
        index = [all_labels[1]]
        numrows = 1
    else:
        index = all_labels
        numrows = len(all_labels)
    #Initialize empty pandas dataframe to store evaluation results across epochs
    #for accuracy, AUROC, and AP
    result_df = pd.DataFrame(data=np.zeros((numrows, num_epochs)),
                            index = index,
                            columns = ['epoch_'+str(n) for n in range(0,num_epochs)])
    #Initialize empty pandas dataframe to store evaluation results for top k
    top_k_result_df = pd.DataFrame(np.zeros((len(all_labels), num_epochs)),
                                   index=[x for x in range(1,len(all_labels)+1)], #e.g. 1,...,64 for len(all_labels)=64
                                   columns = ['epoch_'+str(n) for n in range(0,num_epochs)])
    
    #Make eval results dictionaries
    eval_results_valid = {'accuracy':copy.deepcopy(result_df),
        'auroc':copy.deepcopy(result_df),
        'avg_precision':copy.deepcopy(result_df)}
    eval_results_test = copy.deepcopy(eval_results_valid)
    return eval_results_valid, eval_results_test

def load_existing_evaluation_dfs(all_labels, num_epochs, path_to_dfs, descriptor):
    """Initialize a test eval_dfs_dict from nothing and read in an existing
    validation set eval_dfs_dict"""
    _, eval_results_test = initialize_evaluation_dfs(all_labels, num_epochs)
    keys = ['accuracy','auroc','avg_precision']
    eval_results_valid = {}
    for key in keys:
        filepath = os.path.join(path_to_dfs,descriptor+'_valid_'+key+'_Table.csv')
        eval_results_valid[key] = pd.read_csv(filepath,header=0,index_col=0)
    return eval_results_valid, eval_results_test

def save(eval_dfs_dict, results_dir, descriptor):
    """Variables
    <eval_dfs_dict> is a dict of pandas dataframes
    <descriptor> is a string"""
    for k in eval_dfs_dict.keys():
        eval_dfs_dict[k].to_csv(os.path.join(results_dir, descriptor+'_'+k+'_Table.csv'))
    
def save_final_summary(eval_dfs_dict, best_valid_epoch, setname, general_results_dir, date_and_descriptor):
    """Save to overall df and print summary of best epoch."""
    final_descriptor = date_and_descriptor+'_epoch'+str(best_valid_epoch) #e.g. 2020-08-10-AwesomeModel_epoch43
    if setname=='valid': print('***Summary for',setname,date_and_descriptor,'***')
    for metricname in list(eval_dfs_dict.keys()):
        #metricnames are 'accuracy', 'auroc', and 'avg_precision'.
        #df holds a particular metric for the particular model we just ran.
        #for accuracy, auroc, and avg_precision, df index is diseases, columns are epochs.
        df = eval_dfs_dict[metricname]
        #all_df tracks results of all models in one giant table.
        #all_df has index of diseases or k value, and columns which are particular models.
        all_df_path = os.path.join(general_results_dir,setname+'_'+metricname+'_all.csv') #e.g. valid_accuracy_all.csv
        if os.path.isfile(all_df_path):
            all_df = pd.read_csv(all_df_path,header=0,index_col=0)
            all_df[final_descriptor] = np.nan
        else: #all_df doesn't exist yet - create it.
            all_df = pd.DataFrame(np.empty((df.shape[0],1)),
                                  index = df.index.values.tolist(),
                                  columns = [final_descriptor])
        #Print off and save results for best_valid_epoch
        if setname=='valid': print('\tEpoch',best_valid_epoch,metricname)
        for label in df.index.values:
            #print off to console
            value = df.at[label,'epoch_'+str(best_valid_epoch)]
            if setname=='valid': print('\t\t',label,':',str( round(value, 3) ))
            #save in all_df
            all_df.at[label,final_descriptor] = value
        all_df.to_csv(all_df_path,header=True,index=True)

def clean_up_output_files(best_valid_epoch, results_dir):
    """Delete output files that aren't from the best epoch"""
    #Delete all the backup parameters (they take a lot of space and you do not
    #need to have them)
    shutil.rmtree(os.path.join(results_dir,'backup'))
    #Delete all the extra output files:
    for subdir in ['curves','pred_probs']:
        #Clean up saved ROC and PR curves
        fullpath = os.path.join(results_dir,subdir)
        if os.path.exists(fullpath): #e.g. there may not be a heatmaps dir for a non-bottleneck model
            allfiles = os.listdir(fullpath)
            for filename in allfiles:
                if str(best_valid_epoch) not in filename:
                    os.remove(os.path.join(fullpath,filename))
    print('Output files all clean')
    
#########################
# Calculation Functions #-------------------------------------------------------
#########################        
def evaluate_all(eval_dfs_dict, epoch, label_meanings,
                 true_labels_array, pred_probs_array):
    """Fill out the pandas dataframes in the dictionary <eval_dfs_dict>
    Metrics calculated are: accuracy, AUC, and average precision.
    
    Variables:
    <eval_dfs_dict> is a dictionary of pandas dataframes that will store the
        performance metrics across epochs.
    <epoch> is an integer indicating which epoch it is
    <true_labels_array>: array of true labels. examples x labels
    <pred_probs_array>: array of predicted probabilities. examples x labels"""
    #Accuracy, AUROC, and AP (iter over labels)
    for label_number in range(len(label_meanings)):
        which_label = label_meanings[label_number] #descriptive string for the label
        true_labels = true_labels_array[:,label_number]
        pred_probs = pred_probs_array[:,label_number]
        pred_labels = (pred_probs>=0.5).astype(dtype='int') #decision threshold of 0.5
        
        #Accuracy and confusion matrix (dependent on decision threshold)
        (eval_dfs_dict['accuracy']).at[which_label, 'epoch_'+str(epoch)] = compute_accuracy(true_labels, pred_labels)
        #confusion_matrix, sensitivity, specificity, ppv, npv = compute_confusion_matrix(true_labels, pred_labels)
        
        #AUROC and AP (sliding across multiple decision thresholds)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true = true_labels,
                                         y_score = pred_probs,
                                         pos_label = 1)
        (eval_dfs_dict['auroc']).at[which_label, 'epoch_'+str(epoch)] = sklearn.metrics.auc(fpr, tpr)
        (eval_dfs_dict['avg_precision']).at[which_label, 'epoch_'+str(epoch)] = sklearn.metrics.average_precision_score(true_labels, pred_probs)
    return eval_dfs_dict

def compute_accuracy(true_labels, labels_pred):
    """Print and save the accuracy of the model on the dataset"""    
    correct = (true_labels == labels_pred)
    correct_sum = correct.sum()
    return (float(correct_sum)/len(true_labels))

######################
# Plotting Functions #----------------------------------------------------------
######################
def plot_roc_curve_multi_class(label_meanings, y_test, y_score, 
                               outdir, setname, epoch):
    """<label_meanings>: list of strings, one for each label
    <y_test>: matrix of ground truth
    <y_score>: matrix of predicted probabilities
    <outdir>: directory to save output file
    <setname>: string e.g. 'train' 'valid' or 'test'
    <epoch>: int for epoch"""
    #Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    n_classes = len(label_meanings)
    lw = 2
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    
    #make order df. (note that roc_auc is a dictionary with ints as keys
    #and AUCs as values. 
    order = pd.DataFrame(np.zeros((n_classes,1)), index = [x for x in range(n_classes)],
                         columns = ['roc_auc'])
    for i in range(n_classes):
        order.at[i,'roc_auc'] = roc_auc[i]
    order = order.sort_values(by='roc_auc',ascending=False)
    
    #Plot all ROC curves
    #Plot in order of the rainbow colors, from highest AUC to lowest AUC
    plt.figure()
    colors_list = ['palevioletred','darkorange','yellowgreen','olive','deepskyblue','royalblue','navy']
    curves_plotted = 0
    for i in order.index.values.tolist()[0:10]: #only plot the top ten so the plot is readable
        color_idx = curves_plotted%len(colors_list) #cycle through the colors list in order of colors
        color = colors_list[color_idx]
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{:5s} (area {:0.2f})'.format(label_meanings[i], roc_auc[i]))
        curves_plotted+=1
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(setname.lower().capitalize()+' ROC Epoch '+str(epoch))
    plt.legend(loc="lower right",prop={'size':6})
    outfilepath = os.path.join(outdir,setname+'_ROC_ep'+str(epoch)+'.pdf')
    plt.savefig(outfilepath)
    plt.close()

def plot_pr_curve_multi_class(label_meanings, y_test, y_score, 
                              outdir, setname, epoch):
    """<label_meanings>: list of strings, one for each label
    <y_test>: matrix of ground truth
    <y_score>: matrix of predicted probabilities
    <outdir>: directory to save output file
    <setname>: string e.g. 'train' 'valid' or 'test'
    <epoch>: int for epoch"""
    #Modified from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    #https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    n_classes = len(label_meanings)
    lw = 2
    
    #make order df.
    order = pd.DataFrame(np.zeros((n_classes,1)), index = [x for x in range(n_classes)],
                         columns = ['prc'])
    for i in range(n_classes):
        order.at[i,'prc'] = sklearn.metrics.average_precision_score(y_test[:,i], y_score[:,i])
    order = order.sort_values(by='prc',ascending=False)
    
    #Plot
    plt.figure()
    colors_list = ['palevioletred','darkorange','yellowgreen','olive','deepskyblue','royalblue','navy']
    curves_plotted = 0
    for i in order.index.values.tolist()[0:10]: #only plot the top ten so the plot is readable
        color_idx = curves_plotted%len(colors_list) #cycle through the colors list in order of colors
        color = colors_list[color_idx]
        average_precision = sklearn.metrics.average_precision_score(y_test[:,i], y_score[:,i])
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test[:,i], y_score[:,i])
        plt.step(recall, precision, color=color, where='post',
                 label='{:5s} (area {:0.2f})'.format(label_meanings[i], average_precision))
        curves_plotted+=1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(setname.lower().capitalize()+' PRC Epoch '+str(epoch))
    plt.legend(loc="lower right",prop={'size':6})
    outfilepath = os.path.join(outdir,setname+'_PR_ep'+str(epoch)+'.pdf')
    plt.savefig(outfilepath)
    plt.close()

def plot_learning_curves(train_loss, valid_loss, results_dir, descriptor):
    """Variables
    <train_loss> and <valid_loss> are numpy arrays with one numerical entry
    for each epoch quanitfying the loss for that epoch."""
    x = np.arange(0,len(train_loss))
    plt.plot(x, train_loss, color='blue', lw=2, label='train')
    plt.plot(x, valid_loss, color='green',lw = 2, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, descriptor+'_Learning_Curves.png'))
    plt.close()
    #save numpy arrays of the losses
    np.save(os.path.join(results_dir,'train_loss.npy'),train_loss)
    np.save(os.path.join(results_dir,'valid_loss.npy'),valid_loss)
