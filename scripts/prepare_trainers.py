import ast
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from timm.data import Mixup
from einops.layers.torch import Rearrange
torch.multiprocessing.set_sharing_strategy('file_system')
from torchviz import make_dot
import helper
import os
import time
import matplotlib.pyplot as plt
import subprocess

# 

class MetricsTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.memory_usage = []  # NVIDIA-SMI reported memory
        self.keep_rates = []
        self.start_time = None
        self.epoch_start_time = time.time()  # Start time for the epoch
        
    def start_iteration(self):
        """Call at the start of each iteration to record start time"""
        self.start_time = time.time()
        torch.cuda.empty_cache()  # This is needed but doesn't use PyTorch memory APIs
        
    def end_iteration(self, iteration, keep_rate):
        """Call at the end of each iteration to record metrics"""
        if self.start_time is None:
            return
            
        # Get nvidia-smi reported memory only
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8')
            # Parse the output to get the memory used
            memory_values = [int(x) for x in result.strip().split('\n')]
            nvidia_mem = memory_values[1] if len(memory_values) > 1 else memory_values[0]  # Use GPU 1 if available
        except (subprocess.SubprocessError, IndexError) as e:
            print(f"Error getting GPU memory: {e}")
            nvidia_mem = 0

        self.memory_usage.append(nvidia_mem)
        self.keep_rates.append(keep_rate)

        print(f"Iteration {iteration} - GPU Memory: {nvidia_mem:.2f}MB, Keep Rate: {keep_rate:.2f}")
        
        self.start_time = None
    
    def report_statistics(self, epoch):
        """Report memory statistics at the end of the epoch"""
        # Calculate execution time
        total_time = time.time() - self.epoch_start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Calculate memory statistics
        if self.memory_usage:
            max_mem = max(self.memory_usage)
            mean_mem = sum(self.memory_usage) / len(self.memory_usage)
            std_mem = (sum((x - mean_mem)**2 for x in self.memory_usage) / len(self.memory_usage))**0.5
            
            # Print statistics
            print("\n" + "="*50)
            print("EPOCH STATISTICS")
            print("="*50)
            print(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            print(f"Max Memory: {max_mem:.2f}MB")
            print(f"Mean Memory: {mean_mem:.2f}MB")
            print(f"Std Dev Memory: {std_mem:.2f}MB")
            
            # Save statistics to file
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            stats_path = os.path.join(self.output_dir, f'memory_stats_epoch_{epoch}.txt')
            with open(stats_path, 'w') as f:
                f.write("="*50 + "\n")
                f.write("EPOCH STATISTICS\n")
                f.write("="*50 + "\n")
                f.write(f"Total Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
                f.write(f"Max Memory: {max_mem:.2f}MB\n")
                f.write(f"Mean Memory: {mean_mem:.2f}MB\n")
                f.write(f"Std Dev Memory: {std_mem:.2f}MB\n\n")
                
                # Add memory statistics by keep rate
                f.write("Memory Usage by Keep Rate:\n")
                
                # Group memory by keep rate (rounded to 2 decimal places)
                keep_rate_to_mem = {}
                for kr, mem in zip(self.keep_rates, self.memory_usage):
                    kr_rounded = round(kr, 2)
                    if kr_rounded not in keep_rate_to_mem:
                        keep_rate_to_mem[kr_rounded] = []
                    keep_rate_to_mem[kr_rounded].append(mem)
                
                # Calculate statistics for each keep rate
                for kr in sorted(keep_rate_to_mem.keys()):
                    mem_values = keep_rate_to_mem[kr]
                    avg_mem = sum(mem_values) / len(mem_values)
                    max_mem_kr = max(mem_values)
                    f.write(f"Keep Rate {kr}: Avg Memory: {avg_mem:.2f}MB, Max Memory: {max_mem_kr:.2f}MB, Samples: {len(mem_values)}\n")
            
            # Save as CSV too for easy parsing
            import pandas as pd
            stats_df = pd.DataFrame({
                'metric': ['total_time_seconds', 'max_memory_mb', 'mean_memory_mb', 'std_memory_mb'],
                'value': [total_time, max_mem, mean_mem, std_mem]
            })
            stats_df.to_csv(os.path.join(self.output_dir, f'memory_stats_epoch_{epoch}.csv'), index=False)
            
            # Also save keep rate specific data
            keep_rate_data = []
            for kr in sorted(keep_rate_to_mem.keys()):
                mem_values = keep_rate_to_mem[kr]
                avg_mem = sum(mem_values) / len(mem_values)
                max_mem_kr = max(mem_values)
                keep_rate_data.append({
                    'keep_rate': kr,
                    'avg_memory_mb': avg_mem,
                    'max_memory_mb': max_mem_kr,
                    'samples': len(mem_values)
                })
            
            if keep_rate_data:
                kr_df = pd.DataFrame(keep_rate_data)
                kr_df.to_csv(os.path.join(self.output_dir, f'memory_by_keep_rate_epoch_{epoch}.csv'), index=False)
        
        # Reset for potential next epoch
        self.memory_usage = []
        self.keep_rates = []
        self.epoch_start_time = time.time()  # Reset timer for next epoch

def train_for_image_one_epoch(rank, epoch, num_epochs,
                              model, defined_loss, data_loader,
                              optimizer, lr_schedule, clip_grad,
                              keep_rate=1, random_keep_rate=False, accum_iter=1):
    device = torch.device("cuda:{}".format(rank))
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, num_epochs)
    count = 0
    model.train()

    metrics_output_dir = "/home/cc/data/surya_varrate_variable"
    metrics_tracker = MetricsTracker(metrics_output_dir)

    if rank == 0:
        print('Starting one epoch... ')

    for it, data in enumerate(metric_logger.log_every(iterable=data_loader, print_freq=20, header=header)):
        images = data[0]
        labels = data[1]
        
        # Get the learning rate based on the current iteration number
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        # Move images and labels to gpu
        images = images.to(device, non_blocking=True)
        labels = labels.type(torch.LongTensor)  # <---- Here (casting)
        labels = labels.to(device, non_blocking=True)

        if rank == 0:
            metrics_tracker.start_iteration()

        # Model forward passes + compute the loss
        fp16_scaler = None
        with torch.cuda.amp.autocast(fp16_scaler is not None):
           # x,keep_rate = model(rank, images, keep_rate, random_keep_rate)  # x: logits right after the fc layer
            x = model(rank, images, keep_rate, random_keep_rate)
            keep_rate = model.module.last_keep_rate if hasattr(model, 'module') else model.last_keep_rate

            loss = defined_loss['classification_loss'](x, labels)

        if rank == 0:
            metrics_tracker.end_iteration(it, keep_rate)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if accum_iter == 1:
            # Update network's parameters
            # Clear
            optimizer.zero_grad()
            param_norms = None
            # Fill - backward pass
            loss.backward()
            if clip_grad:
                param_norms = helper.clip_gradients(model, clip_grad)
            # Use
            optimizer.step()

            # Logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            count += 1
        else:
            # Fill - backward pass
            loss.backward()

            # Weights update
            if ((it + 1) % accum_iter == 0) or (it + 1 == len(data_loader)):
                if clip_grad:
                    param_norms = helper.clip_gradients(model, clip_grad)

                # Use
                optimizer.step()

                # Clear
                optimizer.zero_grad()

                # Logging
                torch.cuda.synchronize()
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
                count += 1

    if rank == 0:
        metrics_tracker.report_statistics(epoch)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    global_avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return global_avg_stats



