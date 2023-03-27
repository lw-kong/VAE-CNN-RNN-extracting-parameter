#!/bin/sh
''''exec python -u -- "$0" "$@" # '''
# vi: syntax=python

"""
run.py

Main script for training or evaluating a PDE-VAE model specified by the input file (JSON format).

Usage:
python run.py input_file.json > out
"""

import os
from shutil import copy2
import json
from types import SimpleNamespace
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import time


def setup(in_file,s0):
    # Load configuration from json
    
    s = s0
    # Some defaults
    if not hasattr(s, 'train'):
        raise NameError("'train' must be set to True for training or False for evaluation.")
    elif s.train == False and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for evaluation.")

    if not hasattr(s, 'bool_load_weights'):
        s.bool_load_weights = not s.train
        warnings.warn("Automatically setting 'bool_load_weights' to " + str(s.bool_load_weights))
    if s.bool_load_weights and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for bool_load_weights.")

    if not hasattr(s, 'freeze_encoder'):
        s.freeze_encoder = False
    elif s.freeze_encoder and not s.bool_load_weights:
        raise ValueError("Freeezing encoder weights requires 'bool_load_weights' set to True with encoder weights loaded from file.")

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

    # Create output folder
    if not os.path.exists(s.OUTFOLDER):
        print("Creating output folder: " + s.OUTFOLDER)
        os.makedirs(s.OUTFOLDER)
    elif s.train and os.listdir(s.OUTFOLDER):
        raise FileExistsError("Output folder " + s.OUTFOLDER + " is not empty.")

    # Make a copy of the configuration file in the output folder
    copy2(in_file, s.OUTFOLDER)

    # Print configuration
    print(s)

    # Import class for dataset type
    dataset = __import__(s.dataset_type, globals(), locals(), ['PDEDataset'])
    s.PDEDataset = dataset.PDEDataset

    # Import selected model from models as PDEModel
    models = __import__('models.' + s.model, globals(), locals(), ['VAE'])
    VAEModel = models.VAE

    # Initialize model
    model = VAEModel(param_size=s.param_size, data_channels=s.data_channels,
                    hidden_channels=s.hidden_channels, 
                    prop_layers=s.prop_layers, prop_noise=s.prop_noise,
                    boundary_cond=s.boundary_cond, param_dropout_prob=s.param_dropout_prob, debug=s.debug)
    
    # Set CUDA device
    s.use_cuda = torch.cuda.is_available()
    #s.use_cuda = False
    if s.use_cuda:
        print("Using cuda device(s): " + str(s.cuda_device))
        torch.cuda.set_device(s.cuda_device)
        model.cuda()
    else:
        warnings.warn("Warning: Using CPU only. This is untested.")

    print("\nModel parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t{:<40}{}".format(name + ":", param.shape))

    return model, s


def train(model, s):
    ### Train model on training set
    print("\nTraining...")

    if s.bool_load_weights: # load model to continue training
        print("Loading model from: " + s.MODELLOAD)
        strict_load = not s.freeze_encoder
        if s.use_cuda:
            state_dict = torch.load(s.MODELLOAD, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            state_dict = torch.load(s.MODELLOAD)
        model.load_state_dict(state_dict, strict=strict_load)

        if s.freeze_encoder: # freeze encoder weights
            print("Freezing weights:")
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder." + name + ":", param.size()))
            for name, param in model.encoder_to_param.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder_to_param." + name + ":", param.size()))
            for name, param in model.encoder_to_logvar.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder_to_logvar." + name + ":", param.size()))

    if s.data_parallel:
        model = nn.DataParallel(model, device_ids=s.cuda_device)


    transform = None

    train_loader = torch.utils.data.DataLoader(
        s.PDEDataset(data_file=s.DATAFILE, transform=transform),
        batch_size=s.batch_size, shuffle=True, num_workers=s.num_workers, pin_memory=True,
        worker_init_fn=lambda _: np.random.seed())

    optimizer = torch.optim.Adam(model.parameters(), lr=s.learning_rate, eps=s.eps)

    model.train()

    writer = SummaryWriter(log_dir=os.path.join(s.OUTFOLDER, 'data'))

    # Initialize training variables
    loss_list = []
    recon_loss_list = []
    mse_list = []
    acc_loss = 0
    acc_recon_loss = 0
    acc_latent_loss = 0
    acc_mse = 0
    best_mse = None
    step = 0
    current_discount_rate = s.discount_rate

    ### Training loop
    flag_loss_diverged = 0
    for epoch in range(1, s.max_epochs+1):
        if flag_loss_diverged == 1:
            break
        if epoch < 21 and epoch % 5 == 0:
            print('Epoch: ' + str(epoch))

        # Introduce a discount rate to favor predicting better in the near future
        current_discount_rate = s.discount_rate * np.exp(-s.rate_decay * (epoch-1)) # discount rate decay every epoch
        #print('discount rate = ' + str(current_discount_rate))
        if current_discount_rate > 0:
            w = torch.tensor(np.exp(-current_discount_rate * np.arange(s.validation_length)).reshape(
                    [s.validation_length] + s.data_dimension * [1]), dtype=torch.float32, device='cuda' if s.use_cuda else 'cpu')
            w = w * s.validation_length/w.sum(dim=0, keepdim=True)
        else:
            w = None

        # Load batch and train
        for data, target, data_params in train_loader: ##
            step += 1

            if s.use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            data = data[:,:,:s.input_length]

            target_start = int(np.random.random() * s.validation_random_length)
            target = target[:,:,target_start:]
            target0 = target[:,:,0]
            target = target[:,:,1:s.validation_length+1]


            # Run model
            output, params, logvar = model(data, target0, predict_length=s.validation_length)

            # Reset gradients
            optimizer.zero_grad()

            # Calculate loss
            if s.data_parallel:
                output = output.cpu()
            recon_loss = F.mse_loss(output * w, target * w) if w is not None else F.mse_loss(output, target)
            if s.param_size > 0:
                latent_loss = s.beta * 0.5 * torch.mean(torch.sum(params * params + logvar.exp() - logvar - 1, dim=-1))
                # sum over parameter dimensions
                # average over batch
            else:
                latent_loss = 0
            loss = recon_loss + latent_loss

            mse = F.mse_loss(output.detach(), target.detach()).item() if w is not None else recon_loss.item()

            loss_list.append(loss.item())
            recon_loss_list.append(recon_loss.item())
            mse_list.append(mse)

            acc_loss += loss.item()
            acc_recon_loss += recon_loss.item()
            acc_latent_loss += latent_loss.item()
            acc_mse += mse

            # Calculate gradients
            loss.backward()

            # Clip gradients
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e0)

            # Update gradients
            optimizer.step()

            # Output every 100 steps
            if step % 100 == 0:
                # Check every 500 steps and save checkpoint if new model is at least 2% better than previous best
                if (step > 1 and step % 500 == 0) and ((best_mse is None) or (acc_mse/100 < 0.98*best_mse)):
                    best_mse = acc_mse/100
                    torch.save(model.state_dict(), os.path.join(s.OUTFOLDER, "best.tar"))
                    print('New Best MSE at Step {}: {:.4f}'.format(step, best_mse))

                # Output losses and weights
                if s.param_size > 0:
                    if step > 1:
                        # Write losses to summary
                        writer.add_scalars('losses',    {'loss': acc_loss/100,
                                                         'recon_loss': acc_recon_loss/100,
                                                         'latent_loss': acc_latent_loss/100,
                                                         'mse': acc_mse/100}, step)

                        acc_loss = 0
                        acc_recon_loss = 0
                        acc_latent_loss = 0
                        acc_mse = 0

                    # Write mean model weights to summary
                    weight_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            weight_dict[name] = param.detach().abs().mean().item()
                    writer.add_scalars('weight_avg', weight_dict, step)

                    
                    
                    # Save current set of extracted latent parameters
                    np.savez(os.path.join(s.OUTFOLDER, "training_params.npz"),  data_params=data_params.numpy(), 
                                                                                params=params.detach().cpu().numpy())
                if step % 500 == 0:
                    print('Train Epoch: {}\nTotal Loss: {:.4f}\tPredict. Loss: {:.4f}\tLatent. Loss: {:.4f}\tRecon./Latent: {:.1f}\n'
                            .format(epoch, loss.item(), recon_loss.item(),latent_loss.item(), recon_loss.item()/(latent_loss.item()+1e-8)))
                if epoch > 500 and loss.item() > 100:
                    flag_loss_diverged = 1
                    print('\nLoss diverged\n')
                    break

        # Export checkpoints and loss history after every s.save_epochs epochs
        if s.save_epochs > 0 and epoch % s.save_epochs == 0:
            #torch.save(model.state_dict(), os.path.join(s.OUTFOLDER, "epoch{:06d}.tar".format(epoch)))
            np.savez(os.path.join(s.OUTFOLDER, "loss.npz"), loss=np.array(loss_list), 
                                                            recon_loss=np.array(recon_loss_list), 
                                                            mse=np.array(mse_list))

    return model


tic = time.time()


in_file = os.path.join("input_files", "Lorenz_2_0_0_train.json")


repeat_num = 6

if not os.path.exists(in_file):
    raise FileNotFoundError("Input file " + in_file + " not found.")

for repeat_i in range(repeat_num):
    with open(in_file) as f:
        s0 = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    s0.OUTFOLDER = s0.OUTFOLDER + str(repeat_i)
    if hasattr(s0, 'bool_load_weights'):
        if s0.bool_load_weights:
            s0.MODELLOAD = s0.MODELLOAD + str(repeat_i)
            s0.MODELLOAD = os.path.join(s0.MODELLOAD,"best.tar")
    model, s = setup(in_file,s0)

    if s.train:
        model = train(model, s)

    toc = time.time() - tic
    print('running time ' + str(toc) + 's')
    print(str((repeat_i+1)/repeat_num) + ' is done\n\n')


