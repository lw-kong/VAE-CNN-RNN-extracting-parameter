#!/bin/sh
''''exec python -u -- "$0" "$@" # '''
# vi: syntax=python

"""
run.py

Main script for training or evaluating a PDE-VAE model specified by the input file (JSON format).

Usage:
python run.py input_file.json > out
"""

'evaluating trained model (the whole VAE CNN-RNN module) on test set'

import os
import json
from types import SimpleNamespace
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipyio

import torch
import torch.nn.functional as F


import time

import inflect
int2ordinal = inflect.engine()


def setup(in_file,s0):
    # Load configuration from json
    #with open(in_file) as f:
    #    s = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    s = s0
    # Some defaults
    if not hasattr(s, 'train'):
        raise NameError("'train' must be set to True for training or False for evaluation.")
    elif s.train == False and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for evaluation.")


    if not hasattr(s, 'freeze_encoder'):
        s.freeze_encoder = False
    elif s.freeze_encoder and not s.restart:
        raise ValueError("Freeezing encoder weights requires 'restart' set to True with encoder weights loaded from file.")

    if not hasattr(s, 'data_parallel'):
        s.data_parallel = False
    if not hasattr(s, 'debug'):
        s.debug = False
    if not hasattr(s, 'discount_rate'):
        s.discount_rate = 0.
    if not hasattr(s, 'rate_decay'):
        s.rate_decay = 0.
    if not hasattr(s, 'param_dropout_prob'):
        s.param_dropout_prob = 0.
    if not hasattr(s, 'prop_noise'):
        s.prop_noise = 0.

    if not hasattr(s, 'boundary_cond'):
        raise NameError("Boundary conditions 'boundary_cond' not set. Options include: 'crop', 'periodic', 'dirichlet0'")
    elif s.boundary_cond == 'crop' and (not hasattr(s, 'input_size') or not hasattr(s, 'training_size')):
        raise NameError("'input_size' or 'training_size' not set for crop boundary conditions.")

    # Create output folder
    '''
    if not os.path.exists(s.OUTFOLDER):
        print("Creating output folder: " + s.OUTFOLDER)
        os.makedirs(s.OUTFOLDER)
    elif s.train and os.listdir(s.OUTFOLDER):
        raise FileExistsError("Output folder " + s.OUTFOLDER + " is not empty.")

    # Make a copy of the configuration file in the output folder
    copy2(in_file, s.OUTFOLDER)
    '''
    # Print configuration
    #print(s)

    # Import class for dataset type
    dataset = __import__(s.dataset_type, globals(), locals(), ['PDEDataset'])
    s.PDEDataset = dataset.PDEDataset

    # Import selected model from models as PDEModel
    models = __import__('models.' + s.model, globals(), locals(), ['PDEAutoEncoder'])
    VAEModel = models.VAE

    # Initialize model
    model = VAEModel(param_size=s.param_size, data_channels=s.data_channels,
                    hidden_channels=s.hidden_channels,
                    prop_layers=s.prop_layers, prop_noise=s.prop_noise,
                    boundary_cond=s.boundary_cond, param_dropout_prob=s.param_dropout_prob, debug=s.debug)
    
    # Set CUDA device
    #s.use_cuda = torch.cuda.is_available()
    s.use_cuda = False
    if s.use_cuda:
        print("Using cuda device(s): " + str(s.cuda_device))
        torch.cuda.set_device(s.cuda_device)
        model.cuda()


    return model, s


def plot_train_loss(loss_filename,train_set_size,batch_size):
    loss_read = np.load(loss_filename)
    loss = loss_read['loss']
    recon_loss = loss_read['recon_loss'] # == MSE
    mse = loss_read['mse']
    step_per_epoch = int(train_set_size/batch_size)
    
    drop_epoch = 2000
    tick_plot_size = 14
    label_plot_size = 17
    
    plt.rc('xtick', labelsize=tick_plot_size) 
    plt.rc('ytick', labelsize=tick_plot_size)    
            

    fig_loss, axs_loss = plt.subplots(2, 2,figsize=(12,9), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    axs_loss[0, 0].plot(mse[::step_per_epoch]) 
    axs_loss[0, 0].set_xlabel('epoch',fontsize=label_plot_size)
    axs_loss[0, 0].set_ylabel('MSE loss',fontsize=label_plot_size)
    
    axs_loss[0, 1].plot(loss[::step_per_epoch] - recon_loss[::step_per_epoch]) 
    axs_loss[0, 1].set_xlabel('epoch',fontsize=label_plot_size)
    axs_loss[0, 1].set_ylabel('KL loss',fontsize=label_plot_size)
    
    axs_loss[1, 0].plot(mse[(drop_epoch*step_per_epoch):len(loss):step_per_epoch])
    axs_loss[1, 0].set_xlabel('epoch - '+str(drop_epoch),fontsize=label_plot_size)
    axs_loss[1, 0].set_ylabel('MSE loss',fontsize=label_plot_size)
    
    axs_loss[1, 1].plot(loss[(drop_epoch*step_per_epoch):len(loss):step_per_epoch]
                        - recon_loss[(drop_epoch*step_per_epoch):len(loss):step_per_epoch])
    axs_loss[1, 1].set_xlabel('epoch - '+str(drop_epoch),fontsize=label_plot_size)
    axs_loss[1, 1].set_ylabel('KL loss',fontsize=label_plot_size) 

def plot_train_loss2(loss_filename,train_set_size,batch_size):
    loss_read = np.load(loss_filename)
    step_per_epoch = int(train_set_size/batch_size)
    
    loss = loss_read['loss']
    loss = loss[::step_per_epoch]
    loss_val = loss_read['recon_loss'] # == MSE
    loss_val = loss_val[::step_per_epoch]
    loss_latent = loss - loss_val
    loss_ratio = loss_val/loss_latent
    
    
    
    drop_epoch = 1000
    tick_plot_size = 14
    label_plot_size = 17
    
    plt.rc('xtick', labelsize=tick_plot_size) 
    plt.rc('ytick', labelsize=tick_plot_size)    
            

    fig_loss, axs_loss = plt.subplots(3, 2,figsize=(12,12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    axs_loss[0, 0].plot(loss_val)
    axs_loss[0, 0].set_xlabel('epoch',fontsize=label_plot_size)
    axs_loss[0, 0].set_ylabel('MSE loss',fontsize=label_plot_size)
    
    axs_loss[0, 1].plot(loss_val[drop_epoch:])
    axs_loss[0, 1].set_xlabel('epoch - '+str(drop_epoch),fontsize=label_plot_size)
    axs_loss[0, 1].set_ylabel('MSE loss',fontsize=label_plot_size)   
    
    
    axs_loss[1, 0].plot(loss_latent) 
    axs_loss[1, 0].set_xlabel('epoch',fontsize=label_plot_size)
    axs_loss[1, 0].set_ylabel('KL loss',fontsize=label_plot_size)    
    
    axs_loss[1, 1].plot(loss_latent[drop_epoch:])
    axs_loss[1, 1].set_xlabel('epoch - '+str(drop_epoch),fontsize=label_plot_size)
    axs_loss[1, 1].set_ylabel('KL loss',fontsize=label_plot_size)  
    

    axs_loss[2, 0].plot(loss_ratio) 
    axs_loss[2, 0].set_xlabel('epoch',fontsize=label_plot_size)
    axs_loss[2, 0].set_ylabel('ratio',fontsize=label_plot_size)    
    
    axs_loss[2, 1].plot(loss_ratio[drop_epoch:])
    axs_loss[2, 1].set_xlabel('epoch - '+str(drop_epoch),fontsize=label_plot_size)
    axs_loss[2, 1].set_ylabel('ratio',fontsize=label_plot_size) 
    
    
    print_mean_ratio_len = 500
    
    print("Last " + str(print_mean_ratio_len) + " epochs have mean ratio " 
          +str(np.mean(  loss_ratio[-print_mean_ratio_len:]  )))
    
    
    
def evaluate(model, s, params_filename="params.npz", rmse_filename="rmse_with_depth.npy"):
    ### Evaluate model on test set
    print("\nEvaluating...")

    #if rmse_filename is not None and os.path.exists(os.path.join(s.OUTFOLDER, rmse_filename)):
    #    raise FileExistsError(rmse_filename + " already exists.")
    #if os.path.exists(os.path.join(s.OUTFOLDER, params_filename)):
    #    raise FileExistsError(params_filename + " already exists.")

    if not s.train:
        print("Loading model from: " + s.MODELLOAD)
        if s.use_cuda:
            state_dict = torch.load(s.MODELLOAD, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            state_dict = torch.load(s.MODELLOAD)
        model.load_state_dict(state_dict)
        

    test_loader = torch.utils.data.DataLoader(
        s.PDEDataset(data_file=s.DATAFILE, transform=None),
        batch_size=s.batch_size, num_workers=s.num_workers, pin_memory=True)

    model.eval()
    torch.set_grad_enabled(False)

    ### Evaluation loop
    loss = 0
    if rmse_filename is not None:
        rmse_with_depth = torch.zeros(s.evaluation_length, device='cuda' if s.use_cuda else 'cpu')
    
    params_list = []
    logvar_list = []
    data_params_list = []
    step = 0
    for data, target, data_params in test_loader:
        step += 1

        if s.use_cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)


        target0 = target[:,:,0]
        target = target[:,:,1:s.evaluation_length+1]




        # Run model
        if s.debug:
            output, params, logvar, _, weights, raw_params = model(data.contiguous(), target0, depth=s.evaluation_length)
        else:
            output, params, logvar = model(data.contiguous(), target0, predict_length=s.evaluation_length)

        data_params = data_params.numpy()
        data_params_list.append(data_params)

        if s.param_size > 0:
            params = params.detach().cpu().numpy()
            params_list.append(params)
            logvar_list.append(logvar.detach().cpu().numpy())

        assert output.shape[2] == s.evaluation_length
        loss += F.mse_loss(output, target).item()

        if rmse_filename is not None:
            rmse_with_depth += torch.sqrt(torch.mean((output - target).transpose(2,1).contiguous()
                                        .view(target.size()[0], s.evaluation_length, -1) ** 2,
                                                 dim=-1)).mean(0)
        # averaged over the epoches
        
    # plot evaluation results
    target_np = target.numpy()
    output_np = output.numpy()
    shape_np = np.shape(target_np)
    std_np = (0.5*logvar).exp().numpy()
    
    #t_step = 2 # Ghost
    # Lya_time ~ 100, Ghost 
    # t_text = '$t$'
    
    t_step = 1
    t_text= 'step'
    predict_horizon_plot_max = 150
    predict_horizon_plot_threshold = 40
    
    predict_horizon_plot_max = 50
    predict_horizon_plot_threshold = 20
    
    plot_dim = 2 #z

    
    rmse_horizon_threshold = 0.05
    rmse_all_np = np.sum((target_np - output_np)**2,axis=1)
    state_lenth_np = np.sum(target_np**2,axis=1)
    rmse_horizon_np = np.zeros(shape_np[0])
    
    for i in range(shape_np[0]):
        for ti in range(shape_np[2]):
            if rmse_all_np[i,ti] > rmse_horizon_threshold * state_lenth_np[i,ti]:
                break
        rmse_horizon_np[i] = ti * t_step
    
    tick_plot_size = 14
    label_plot_size = 17 
    plt.rc('xtick', labelsize=tick_plot_size) 
    plt.rc('ytick', labelsize=tick_plot_size) 

    ###
    # plot about RNN prediction
    fig_predict, axs_predict = plt.subplots(4, 2,figsize=(11,16), gridspec_kw={'hspace': 0.28, 'wspace': 0.28})    
    plot_trial = 2 # 0.9944
    axs_predict[0, 0].plot(t_step*np.arange(shape_np[2]),target_np[plot_trial,plot_dim,:],'b')
    axs_predict[0, 0].plot(t_step*np.arange(shape_np[2]),output_np[plot_trial,plot_dim,:],'r--')
    plot_trial = 11 #0.9924
    axs_predict[0, 1].plot(t_step*np.arange(shape_np[2]),target_np[plot_trial,plot_dim,:],'b')
    axs_predict[0, 1].plot(t_step*np.arange(shape_np[2]),output_np[plot_trial,plot_dim,:],'r--')    
    plot_trial = np.random.randint(0,shape_np[0]) 
    axs_predict[1, 0].plot(t_step*np.arange(shape_np[2]),target_np[plot_trial,plot_dim,:],'b')
    axs_predict[1, 0].plot(t_step*np.arange(shape_np[2]),output_np[plot_trial,plot_dim,:],'r--')    
    plot_trial = np.random.randint(0,shape_np[0]) 
    axs_predict[1, 1].plot(t_step*np.arange(shape_np[2]),target_np[plot_trial,plot_dim,:],'b')
    axs_predict[1, 1].plot(t_step*np.arange(shape_np[2]),output_np[plot_trial,plot_dim,:],'r--')
    plot_trial = np.random.randint(0,shape_np[0]) 
    axs_predict[2, 0].plot(t_step*np.arange(shape_np[2]),target_np[plot_trial,plot_dim,:],'b')
    axs_predict[2, 0].plot(t_step*np.arange(shape_np[2]),output_np[plot_trial,plot_dim,:],'r--')    
    plot_trial = np.random.randint(0,shape_np[0]) 
    axs_predict[2, 1].plot(t_step*np.arange(shape_np[2]),target_np[plot_trial,plot_dim,:],'b')
    axs_predict[2, 1].plot(t_step*np.arange(shape_np[2]),output_np[plot_trial,plot_dim,:],'r--')  
    for ax in axs_predict.flat:
        ax.set_xlabel(t_text,fontsize=label_plot_size)
        ax.set_ylabel('$P$',fontsize=label_plot_size)
    axs_predict[3, 0].scatter(data_params,rmse_horizon_np)
    axs_predict[3, 0].plot(data_params,predict_horizon_plot_threshold*np.ones_like(data_params),'r--')
    axs_predict[3, 0].set_ylabel('Prediction Horizon',fontsize=label_plot_size)
    axs_predict[3, 0].set_ylim(bottom=0, top=predict_horizon_plot_max)
    axs_predict[3, 0].set_xlabel('Real Hidden Parameter',fontsize=label_plot_size)
    axs_predict[3, 1].hist(rmse_horizon_np, weights=np.zeros_like(rmse_horizon_np) + 1. / len(rmse_horizon_np))
    axs_predict[3, 1].set_ylabel('Frequency',fontsize=label_plot_size)
    axs_predict[3, 1].set_xlabel('Prediction Horizon',fontsize=label_plot_size)
        
        
    
    ###
    # plot about hidden variables        
    fig_para, axs_para = plt.subplots(2, 1,figsize=(8,9), gridspec_kw={'hspace': 0.25})
    axs_para[0].bar(range(1,s.param_size+1),np.var(params,axis = 0))  
    axs_para[0].set_ylabel('var($\mu_k^{(i)}$)',fontsize=label_plot_size)
    axs_para[1].bar(range(1,s.param_size+1),np.mean(std_np,axis = 0),color='r') 
    axs_para[1].set_ylabel('mean($\sigma_k^{(i)}$)',fontsize=label_plot_size)
    axs_para[1].invert_yaxis()
    for ax in axs_para.flat:
        ax.set_xlabel('dimension',fontsize=label_plot_size)        


    errorbar_linewidth = 0.2
    errorbar_color = 'green'
    
    subplot_y_num = 3
    subplot_x_num = 2
    fig_para, axs_para = plt.subplots(subplot_y_num, subplot_x_num,
                                      figsize=(10,15), gridspec_kw={'hspace': 0.35, 'wspace': 0.35})    
    for d_i in range(s.param_size):
        subplot_i = int(d_i/subplot_x_num)
        subplot_j = np.mod(d_i,subplot_x_num)
        axs_para[subplot_i, subplot_j].errorbar(data_params,params[:,d_i],std_np[:,d_i],
                                ecolor=errorbar_color,elinewidth=errorbar_linewidth, ls='none') 
        axs_para[subplot_i, subplot_j].scatter(data_params,params[:,d_i])
        axs_para[subplot_i, subplot_j].set_title(int2ordinal.ordinal(d_i+1) + ' dimension', fontsize=label_plot_size)
    for ax in axs_para.flat:
        ax.set_xlabel('real',fontsize=label_plot_size)
        ax.set_ylabel('predicted',fontsize=label_plot_size)
    #fig_para.savefig("fig_para.pdf", bbox_inches='tight')
    
    return params,std_np

tic = time.time()

# Not using CUDA

#in_file = os.path.join("input_files", "Ghost_2_0_eva_repeat.json")
in_file = os.path.join("input_files", "Lorenz_2_eva_repeat.json")
#in_file = os.path.join("input_files", "Lorenz_2_eva_outside_repeat.json")
repeat_num = 4

param_all = np.zeros([repeat_num,500,5])
std_all = np.zeros([repeat_num,500,5])
if not os.path.exists(in_file):
    raise FileNotFoundError("Input file " + in_file + " not found.")

for repeat_i in range(repeat_num):
    with open(in_file) as f:
        s0 = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    load_foldername = s0.MODELLOAD + str(repeat_i)
    s0.MODELLOAD =  os.path.join(load_foldername,"best.tar")
    filename_loss = os.path.join(load_foldername,"loss.npz")
    
    plot_train_loss2(filename_loss,1000,100)

for repeat_i in range(repeat_num):
    with open(in_file) as f:
        s0 = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    load_foldername = s0.MODELLOAD + str(repeat_i)
    s0.MODELLOAD =  os.path.join(load_foldername,"best.tar")
    filename_loss = os.path.join(load_foldername,"loss.npz")
    
    model, s = setup(in_file,s0)
    params,stds = evaluate(model, s)
    param_all[repeat_i,:,:] = params
    std_all[repeat_i,:,:] = stds
    
toc = time.time() - tic
print('running time ' + str(toc) + 's')

'''
run_name = "Lorenz_2_0_0_step6000"
save_dic = {"para_VAE_all": param_all,"std_VAE_all": std_all, "run_name": run_name , "test_data": s.DATAFILE }
scipyio.savemat(os.path.join("output_data","save_" + run_name +".mat"), save_dic)
'''
