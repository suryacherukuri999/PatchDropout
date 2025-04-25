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

class MetricsTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.memory_usage = []
        self.cpu_memory_usage = []  # Added for CPU memory tracking
        self.iteration_times = []
        self.iterations = []
        self.keep_rates = []
        self.MB = 1024.0 * 1024.0
        self.start_time = None
        
    def start_iteration(self):
        """Call at the start of each iteration to record start time"""
        self.start_time = time.time()
        #torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for this iteration
        torch.cuda.empty_cache()
        
    def end_iteration(self, iteration, keep_rate):
        """Call at the end of each iteration to record metrics"""
        if self.start_time is None:
            return
            
        # Record iteration time
        iter_time = time.time() - self.start_time
        self.iteration_times.append(iter_time)
        
        # Record GPU memory usage
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8')
            # Parse the output to get the memory used for GPU 1
            memory_values = [int(x) for x in result.strip().split('\n')]
            memory_used = memory_values[1] if len(memory_values) > 1 else memory_values[0]  # Use GPU 1 if available
        except (subprocess.SubprocessError, IndexError) as e:
            print(f"Error getting GPU memory: {e}")
            memory_used = 0

        self.memory_usage.append(memory_used)
        
        # Record CPU memory usage - fixed implementation
        try:
            # Get process ID of our Python process
            import os
            pid = os.getpid()
            print("surya process" + str(pid))
            # The '=' after 'rss' ensures that no header is printed.
            result = subprocess.check_output(
                ['ps', '-p', str(pid), '-o', 'rss='],
                encoding='utf-8')
            mem_kb = float(result.strip())
            cpu_memory_used = mem_kb / 1024.0  # Convert from KB to MB
            
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            print(f"Error getting CPU memory: {e}")
            cpu_memory_used = 0
            
        self.cpu_memory_usage.append(cpu_memory_used)
        
        # Record iteration number and keep rate
        self.iterations.append(iteration)
        self.keep_rates.append(keep_rate)

        print(f"Iteration {iteration} - Time: {iter_time:.4f}s, GPU Memory: {memory_used:.2f}MB, CPU Memory: {cpu_memory_used:.2f}MB, Keep Rate: {keep_rate:.2f}")

        # Save metrics to CSV after each iteration to ensure data is not lost
        self.save_metrics_to_csv(iteration)
        
        self.start_time = None
        
    def save_metrics_to_csv(self, current_iteration):
        """Save metrics to CSV file after each iteration"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create a DataFrame with just the latest metrics
        import pandas as pd
        latest_df = pd.DataFrame({
            'iteration': [self.iterations[-1]],
            'gpu_memory_mb': [self.memory_usage[-1]],
            'cpu_memory_mb': [self.cpu_memory_usage[-1]],  
            'time_seconds': [self.iteration_times[-1]],
            'keep_rate': [self.keep_rates[-1]]
        })
        
        # Define the CSV file path
        csv_path = os.path.join(self.output_dir, f'metrics_realtime.csv')
        
        # If file doesn't exist, create it with header
        # If it exists, append without header
        if not os.path.exists(csv_path):
            latest_df.to_csv(csv_path, index=False)
        else:
            latest_df.to_csv(csv_path, mode='a', header=False, index=False)
        
    def plot_metrics(self, epoch):
        """Generate plots for the collected metrics"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Plot GPU memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.memory_usage)
        plt.title(f'GPU Memory Usage - Epoch {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('GPU Memory Used (MB)')
        plt.xlim(0, max(self.iterations) if self.iterations else 100)  # Start x-axis from 0
        plt.ylim(0, max(self.memory_usage) * 1.1 if self.memory_usage else 1000)  # Start y-axis from 0
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'gpu_memory_usage_epoch_{epoch}.png'))
        plt.close()
        
        # Plot CPU memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.cpu_memory_usage)
        plt.title(f'CPU Memory Usage - Epoch {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('CPU Memory Used (MB)')
        plt.xlim(0, max(self.iterations) if self.iterations else 100)  # Start x-axis from 0
        plt.ylim(0, max(self.cpu_memory_usage) * 1.1 if self.cpu_memory_usage else 1000)  # Start y-axis from 0
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'cpu_memory_usage_epoch_{epoch}.png'))
        plt.close()
        
        # Plot iteration times
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.iteration_times)
        plt.title(f'Iteration Times - Epoch {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        plt.xlim(0, max(self.iterations) if self.iterations else 100)  # Start x-axis from 0
        plt.ylim(0, max(self.iteration_times) * 1.1 if self.iteration_times else 10)  # Start y-axis from 0
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'iteration_times_epoch_{epoch}.png'))
        plt.close()
        
        # Plot keep rates
        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.keep_rates)
        plt.title(f'Keep Rates - Epoch {epoch}')
        plt.xlabel('Iteration')
        plt.ylabel('Keep Rate')
        plt.xlim(0, max(self.iterations) if self.iterations else 100)  # Start x-axis from 0
        plt.ylim(0, 1.1)  # Keep rate is between 0 and 1
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'keep_rates_epoch_{epoch}.png'))
        plt.close()
        
        # Plot memory vs keep rate scatter for GPU
        plt.figure(figsize=(10, 6))
        plt.scatter(self.keep_rates, self.memory_usage, alpha=0.6)
        plt.title(f'GPU Memory Usage vs Keep Rate - Epoch {epoch}')
        plt.xlabel('Keep Rate')
        plt.ylabel('GPU Memory Used (MB)')
        plt.xlim(0, 1.1)  # Keep rate is between 0 and 1
        plt.ylim(0, max(self.memory_usage) * 1.1 if self.memory_usage else 1000)  # Start y-axis from 0
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'gpu_memory_vs_keep_rate_epoch_{epoch}.png'))
        plt.close()
        
        # Plot memory vs keep rate scatter for CPU
        plt.figure(figsize=(10, 6))
        plt.scatter(self.keep_rates, self.cpu_memory_usage, alpha=0.6)
        plt.title(f'CPU Memory Usage vs Keep Rate - Epoch {epoch}')
        plt.xlabel('Keep Rate')
        plt.ylabel('CPU Memory Used (MB)')
        plt.xlim(0, 1.1)  # Keep rate is between 0 and 1
        plt.ylim(0, max(self.cpu_memory_usage) * 1.1 if self.cpu_memory_usage else 1000)  # Start y-axis from 0
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'cpu_memory_vs_keep_rate_epoch_{epoch}.png'))
        plt.close()
        
        # Save all metrics to epoch-specific CSV
        import pandas as pd
        df = pd.DataFrame({
            'iteration': self.iterations,
            'gpu_memory_mb': self.memory_usage,
            'cpu_memory_mb': self.cpu_memory_usage,
            'time_seconds': self.iteration_times,
            'keep_rate': self.keep_rates
        })
        df.to_csv(os.path.join(self.output_dir, f'metrics_epoch_{epoch}.csv'), index=False)
        
        # Reset for next epoch
        self.memory_usage = []
        self.cpu_memory_usage = []
        self.iteration_times = []
        self.iterations = []
        self.keep_rates = []

def train_for_image_one_epoch(rank, epoch, num_epochs,
                              model, defined_loss, data_loader,
                              optimizer, lr_schedule, clip_grad,
                              keep_rate=1, random_keep_rate=False, accum_iter=1):
    device = torch.device("cuda:{}".format(rank))
    metric_logger = helper.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, num_epochs)
    count = 0
    model.train()

    metrics_output_dir = "/home/saisurya/data/surya_varrate_variable"
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
            x,keep_rate = model(rank, images, keep_rate, random_keep_rate)  # x: logits right after the fc layer
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
        metrics_tracker.plot_metrics(epoch)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    global_avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return global_avg_stats



