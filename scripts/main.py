import os
import sys
import math
import time
import shutil
import datetime
from datetime import timedelta
import yaml
import json
import time
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from easydict import EasyDict as edict

import torch
import torchtext
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
import torch.nn.functional as F
import timm
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import helper
import prepare_augmentations
import prepare_datasets
import prepare_models
import prepare_trainers
import evaluation

import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

def count_model_parameters_breakdown(model):
    """
    Calculate detailed statistics about model parameters including
    separate counts for weights and biases, and memory usage.
    
    Args:
        model: PyTorch model (can be wrapped in DDP)
    
    Returns:
        dict: Parameter statistics including counts and memory usage
    """
    # Check if model is wrapped in DDP
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    
    total_params = 0
    weight_params = 0
    bias_params = 0
    other_params = 0
    
    # Print a header for the table
    print(f"{'Layer':<40} {'Shape':<25} {'Parameters':<12} {'Type':<8}")
    print("-" * 85)
    
    # Iterate through each named parameter
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue  # Skip frozen parameters
            
        params = parameter.numel()  # Number of elements in the parameter tensor
        shape_str = str(list(parameter.shape))
        
        # Determine parameter type
        if 'weight' in name:
            param_type = 'Weight'
            weight_params += params
        elif 'bias' in name:
            param_type = 'Bias'
            bias_params += params
        else:
            param_type = 'Other'
            other_params += params
            
        print(f"{name:<40} {shape_str:<25} {params:<12,} {param_type:<8}")
        total_params += params
    
    # Calculate memory size (assuming 4 bytes per parameter for float32)
    bytes_per_param = 4  # float32 size
    total_memory = total_params * bytes_per_param
    total_memory_mb = total_memory / (1024 * 1024)
    weight_memory_mb = weight_params * bytes_per_param / (1024 * 1024)
    bias_memory_mb = bias_params * bytes_per_param / (1024 * 1024)
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 85)
    print(f"Total parameters:    {total_params:,}")
    print(f"Weight parameters:   {weight_params:,} ({weight_params/total_params*100:.2f}%)")
    print(f"Bias parameters:     {bias_params:,} ({bias_params/total_params*100:.2f}%)")
    print(f"Other parameters:    {other_params:,} ({other_params/total_params*100:.2f}%)")
    print(f"Memory usage:        {total_memory:,} bytes")
    print(f"                     {total_memory / 1024:,.2f} KB")
    print(f"                     {total_memory_mb:.2f} MB")
    print(f"Weight memory:       {weight_memory_mb:.2f} MB ({weight_memory_mb/total_memory_mb*100:.2f}%)")
    print(f"Bias memory:         {bias_memory_mb:.2f} MB ({bias_memory_mb/total_memory_mb*100:.2f}%)")
    
    return {
        'total_params': total_params,
        'weight_params': weight_params,
        'bias_params': bias_params,
        'other_params': other_params,
        'total_memory_mb': total_memory_mb,
        'weight_memory_mb': weight_memory_mb,
        'bias_memory_mb': bias_memory_mb
    }

def prepare_args(dataset_params_path, default_params_path=None):
    # Load default params
    with open(default_params_path) as f:
        args = yaml.safe_load(f)

    # Load dataset params
    with open(dataset_params_path) as f:
        dataset_params_args = yaml.safe_load(f)

    # Combine default and dataset params
    args['dataset_params']['dataset_choice'] = dataset_params_args['dataset_params']['dataset_choice']

    if args['mode'] == 'inference':
        args['training_params']['inference'] = args['training_params']['train']
        del args['training_params']['train']

    return args


def set_globals(rank, args):
    global mode
    global save_params, dataset_params, system_params, dataloader_params, model_params
    global training_params, trainloader_params, valloader_params

    # Prepare mode, and other "_params"
    mode = args['mode']
    save_params = args['save_params']
    dataset_params = args['dataset_params']
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']
    model_params = args['model_params']
    training_params = args['training_params']
    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']

    # ============ Writing model_path in args ... ============
    # Write model_path in args
    output_dir = save_params["output_dir"]
    model_path = os.path.join(output_dir, f"PatchDropout_{dataset_params['dataset_name']}_{model_params['backbone_option']}")
    if not os.path.exists(model_path):
        if rank == 0:
            os.makedirs(model_path)
    args["save_params"]["model_path"] = model_path
    return args


def get_data(rank):
    global TimmModels
    # Get augmentations
    augmentations = prepare_augmentations.PublicDataAugmentation(dataset_params)
    transforms_plain = augmentations.transforms_plain
    transforms_aug = augmentations.transforms_aug
    normalize = augmentations.normalize

    # Get dataset class
    backbone_option = model_params['backbone_option']
    dataset_name = dataset_params['dataset_name']
    num_classes = dataset_params['dataset_choice'][dataset_name]['num_classes']
    pretrained = True

    TimmModels = prepare_models.TimmModels(backbone_option,
                                           pretrained=pretrained,
                                           num_classes=num_classes,
                                           img_size=int(dataset_params['resolution']))
    model_params['patch_size'] = TimmModels.model.patch_size

    if dataset_name != 'CSAW':
        dataset_class = prepare_datasets.GetPublicDatasets(dataset_params, transforms_aug=transforms_aug, transforms_plain=transforms_plain, normalize=normalize)

        if mode == 'train':
            train_dataset, val_dataset = dataset_class.get_datasets('train/')
            train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=system_params['num_gpus'], rank=rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            val_dataset = dataset_class.get_datasets('test/')

        # Set the data sampler with DistributedSampler
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=system_params['num_gpus'], rank=rank, shuffle=True)  # shuffle=True to reduce monitor bias

    else:
        dataset_class = prepare_datasets.GetCSAWDatasets(dataset_params,
                                                         normalize=normalize,
                                                         patch_size=model_params['patch_size'],
                                                         input_type_name=training_params[mode]['input_type']['name'])

        if mode == 'train':
            train_dataset = dataset_class.get_datasets('train/', transforms_aug)
            val_dataset = dataset_class.get_datasets('val/', transforms_plain)
        else:
            val_dataset = dataset_class.get_datasets('test/', transforms_plain)

        # Set the training sampler
        # We make sure that the classes are balanced and for inputs with only breast tokens,
        # We group training samples according to their lengths
        if mode == 'train':  # if train
            train_SimilarLengthBalancedClassSampler = prepare_datasets.SimilarLengthBalancedClassSampler(
                                                                                                        train_dataset,
                                                                                                        similar_length_column=None,
                                                                                                        if_similar_length=False,
                                                                                                        if_balanced_class=True)
            train_sampler = prepare_datasets.DistributedSamplerWrapper(sampler=train_SimilarLengthBalancedClassSampler, shuffle=False)

        # Set the validation sampler with DistributedSampler
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=system_params['num_gpus'], rank=rank,
                                                          shuffle=True)  # shuffle=True to reduce monitor bias

    # Build train and val data loaders
    train_batch_size = int(trainloader_params['batch_size'] / (system_params['num_gpus']*trainloader_params['accum_iter']))
    val_batch_size = int(valloader_params['batch_size'] / (system_params['num_gpus']*valloader_params['accum_iter']))
    if mode == 'train':  # if train
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=train_batch_size,
                                      num_workers=trainloader_params['num_workers'],
                                      pin_memory=trainloader_params['pin_memory'],
                                      drop_last=trainloader_params['drop_last'])
    else:
        train_dataloader = None
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler,
                                batch_size=val_batch_size,
                                num_workers=valloader_params['num_workers'],
                                pin_memory=valloader_params['pin_memory'],
                                drop_last=valloader_params['drop_last'])

    if rank == 0 and mode == 'train':
        print(f"There are {len(train_dataloader)} train_dataloaders on each rank. ")
        print(f"There are {len(val_dataloader)} val_dataloaders on each rank. ", end="\n\n")

    return rank, train_dataloader, val_dataloader


def get_model_loss(rank):
    # ============ Preparing model ... ============
    model = TimmModels.model

    # Move the model to gpu. This step is necessary for DDP later
    device = torch.device("cuda:{}".format(rank))
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    # Log the number of trainable parameters in Tensorboard
    if rank == 0:
        param_stats = count_model_parameters_breakdown(model)
        print(f"TOTAL MODEL SIZE: {param_stats['total_memory_mb']:.2f} MB")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of trainable params in the model:', n_parameters, end='\n\n')

    # ============ Preparing loss and move it to gpu ... ============
    loss = torch.nn.CrossEntropyLoss(label_smoothing=training_params[mode]['label_smoothing'])
    return rank, {'model': model}, {'classification_loss': loss}


def get_optimizer(model):
    # ============ Preparing optimizer ... ============
    try:
        skip_weight_decay_list = model['model'].module.encoder.no_weight_decay()
    except:
        skip_weight_decay_list = ()

    params_dict = helper.layer_decay_get_params_groups(model, weight_decay=training_params[mode]['wd'],
                                                       skip_list=skip_weight_decay_list,
                                                       get_num_layer=None,
                                                       get_layer_scale=None)

    optimizer_choice = training_params[mode]['optimizer']['name']
    lr = training_params[mode]['optimizer'][optimizer_choice]['lr']
    momentum = training_params[mode]['optimizer'][optimizer_choice]['momentum']
    if optimizer_choice == "adamw":
        optimizer = torch.optim.AdamW(params_dict, lr=lr, betas=(momentum[0], momentum[1]))
    elif optimizer_choice == "sgd":
        optimizer = torch.optim.SGD(params_dict, lr=lr, momentum=momentum)
    return optimizer


def get_schedulers(train_dataloader):
    # ============ Initialize schedulers ... ============
    optimizer_choice = training_params[mode]['optimizer']['name']
    base_lr = training_params[mode]['optimizer'][optimizer_choice]['lr']

    if 'warmup_epochs' not in training_params[mode]['optimizer'][optimizer_choice].keys():
        warmup_epochs = 0
        step_epoch = 1000
    else:
        warmup_epochs = training_params[mode]['optimizer'][optimizer_choice]['warmup_epochs']
        step_epoch = training_params[mode]['optimizer'][optimizer_choice]['step_epoch']

    lr_schedule = helper.constant_scheduler(base_value=base_lr, epochs=training_params[mode]['num_epochs'],
                                            niter_per_ep=len(train_dataloader), warmup_epochs=warmup_epochs,
                                            start_warmup_value=0, step_epoch=step_epoch)
    return lr_schedule


def train_process(rank, train_dataloader, val_dataloader, model, loss, optimizer, lr_schedule):
    print('Training timm model from imagenet pretrained')

    model = model['model']
    if training_params[mode]['input_type']['name'] == 'SampledTokens':
        keep_rate = training_params[mode]['input_type']['SampledTokens']['keep_rate']
        if isinstance(keep_rate, str):
            keep_rate = float(eval(keep_rate))
        else:
            keep_rate = float(keep_rate)
    else:
        keep_rate = 1
    random_keep_rate = training_params[mode]['input_type']['SampledTokens']['random_keep_rate']

    # # ============ Optionally resume training ... ============
    to_restore = {'epoch': 0}
    helper.restart_from_checkpoint(
        os.path.join(save_params['model_path'], f"checkpoint_{dataset_params['dataset_name']}_ImagenetSupervised.pth"),
        run_variables=to_restore, model=model, optimizer=optimizer)

    # # ============ Start training ... ============
    if rank == 0:
        print("Starting training !")

    for epoch in range(to_restore['epoch'], training_params[mode]['num_epochs']):
        # In distributed mode, calling the :meth:`set_epoch` method at
        # the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        # is necessary to make shuffling work properly across multiple epochs. Otherwise,
        # the same ordering will be always used.
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        # ============ Training one epoch of finetuning ... ============
        _ = prepare_trainers.train_for_image_one_epoch( rank, epoch, training_params[mode]['num_epochs'],
                                                        model, loss, train_dataloader,
                                                        optimizer, lr_schedule, training_params[mode]['clip_grad'],
                                                        keep_rate=keep_rate, random_keep_rate=random_keep_rate,
                                                        accum_iter=trainloader_params['accum_iter'])

        # ============ Evaluating the classification performance ... ============
        if dataset_params['dataset_name'] != 'CSAW':
            if (epoch + 1) % training_params[mode]['val_freq'] == 0 or (epoch + 1) == training_params[mode]['num_epochs']:
                val_loss, acc1, acc5 = evaluation.public_validate_network(rank, val_dataloader, model, dataset_params, keep_rate=1, random_keep_rate=False)
                if rank == 0:
                    print(f"Val Loss at epoch {epoch+1} of the network on the validation set: {val_loss}")
                    print(f"Top1 accuracy at epoch {epoch+1} of the network on the validation set: {acc1}")
                    print(f"Top5 accuracy at epoch {epoch+1} of the network on the validation set: {acc5}")
        else:
            if (epoch + 1) % training_params[mode]['val_freq'] == 0 or (epoch + 1) == training_params[mode]['num_epochs']:
                val_loss, img_auc, avg_auc, max_auc, min_auc = evaluation.csaw_validate_for_image_network(rank, val_dataloader, model, keep_rate=1, random_keep_rate=False)
                if rank == 0:
                    print(f"Val Loss at epoch {epoch+1} of the network on the validation set: {val_loss}")
                    print(f"Image AUC at epoch {epoch+1} of the network on the validation set: {img_auc}")
                    print(f"Exam-avg AUC at epoch {epoch+1} of the network on the validation set: {avg_auc}")
                    print(f"Exam-max AUC at epoch {epoch+1} of the network on the validation set: {max_auc}")
                    print(f"Exam-min AUC at epoch {epoch+1} of the network on the validation set: {min_auc}")

        # ============ Saving the model ... ============
        save_dict = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}

        if rank == 0:
            torch.save(save_dict,
                       os.path.join(save_params['model_path'], f"checkpoint_{dataset_params['dataset_name']}_ImagenetSupervised.pth"))
            if save_params['saveckp_freq'] and (epoch + 1) % save_params['saveckp_freq'] == 0:
                torch.save(save_dict, os.path.join(save_params['model_path'],
                                                   f"checkpoint{epoch + 1:04}_{dataset_params['dataset_name']}_ImagenetSupervised.pth"))
    print('Training finished!')

    return None

def main(rank, args):
    # ============ Define some parameters for easy access... ============
    _ = set_globals(rank, args)

    # ============ Setting up system configuration ... ============
    # Set gpu params and random seeds for reproducibility
    helper.set_sys_params(system_params['random_seed'])

    # ============ Getting data ready ... ============
    rank, train_dataloader, val_dataloader = get_data(rank)

    # ============ Getting model and loss ready ... ============
    rank, model, loss = get_model_loss(rank)

    if mode == 'train':
        # ============ Getting optimizer ready ... ============
        optimizer = get_optimizer(model)

        # ============ Getting schedulers ready ... ============
        lr_schedule = get_schedulers(train_dataloader)

        if rank == 0:
            print(f"Loss, optimizer and schedulers ready.", end="\n\n")

        # ============ Start training process ... ============
        train_process(rank, train_dataloader, val_dataloader, model, loss, optimizer, lr_schedule)


if __name__ == '__main__':
    args = prepare_args(dataset_params_path='yaml/dataset/dataset_params.yaml',
                        default_params_path='yaml/train_yaml/default_params.yaml')
    helper.launch(main, args)



