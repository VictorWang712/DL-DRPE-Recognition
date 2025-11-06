import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dataset import DRPESiameseDataset
from siamese_network_vit_clip import DualViTCLIP
from info_nce_loss import InfoNCELoss
import torchvision.transforms as transforms
import random
import numpy as np
import argparse
from datetime import datetime
import json

def set_seed(seed=42):  # Set random seed for reproducibility 
    random.seed(seed)  # Set Python random seed 
    np.random.seed(seed)  # Set NumPy random seed 
    torch.manual_seed(seed)  # Set PyTorch CPU seed 
    torch.cuda.manual_seed_all(seed)  # Set PyTorch CUDA seed 

def main_worker(local_rank, world_size, args):  # Main worker for each process 
    set_seed()  # Ensure deterministic behavior 
    torch.cuda.set_device(local_rank)  # Set current CUDA device 
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)  # Initialize distributed training 

    grey_root = '../../data/grey'  # Path to grayscale images 
    encrypted_root = '../../data/drpe_encrypted'  # Path to encrypted images 
    batch_size = args.batch_size  # Batch size per GPU 
    num_epochs = args.epochs  # Number of training epochs 
    lr = args.lr  # Learning rate 
    embedding_dim = args.embedding_dim  # Embedding dimension 
    vit_variant = args.vit_variant  # ViT variant to use 

    base_model_dir = '../../model'  # Directory to save models 
    os.makedirs(base_model_dir, exist_ok=True)  # Create directory if not exists 
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Timestamp for this run 
    model_subdir = os.path.join(base_model_dir, f'run_{run_time}')  # Subdirectory for this run 
    os.makedirs(model_subdir, exist_ok=True)  # Create run subdirectory 

    log_dict = {  # Dictionary to store training logs 
        "start_time": run_time,  # Training start time 
        "args": vars(args),  # Training arguments 
        "model_subdir": model_subdir,  # Model directory 
        "epoch_logs": []  # List to store per-epoch logs 
    }
    log_path = os.path.join(model_subdir, "train_log.json")  # Path to save log file 

    transform = transforms.Compose([  # Compose image transformations 
        transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT 
        transforms.ToTensor(),  # Convert images to tensor 
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images 
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)  # Create dataset 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)  # Distributed sampler for data 
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)  # DataLoader with distributed sampler 

    model = DualViTCLIP(  # Instantiate dual ViT-CLIP model 
        embedding_dim=embedding_dim,  # Set embedding dimension 
        vit_variant=vit_variant,  # Set ViT variant 
        pretrained=True  # Use pretrained weights 
    ).cuda(local_rank)  # Move model to current GPU 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # Wrap model for DDP 
    criterion = InfoNCELoss(temperature=0.07).cuda(local_rank)  # InfoNCE loss for contrastive learning 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # AdamW optimizer 

    for epoch in range(num_epochs):  # Loop over epochs 
        model.train()  # Set model to training mode 
        sampler.set_epoch(epoch)  # Shuffle data differently at each epoch 
        total_loss = 0  # Track total loss for this epoch 
        num_batches = 0  # Track number of batches 
        for grey_img, enc_img, _ in tqdm(  # Iterate over batches 
            dataloader,
            desc=f"[GPU {local_rank}] Epoch {epoch + 1}/{num_epochs}",  # Progress bar description 
            disable=(local_rank != 0)  # Only show progress bar on rank 0 
        ):
            grey_img = grey_img.cuda(local_rank, non_blocking=True)  # Move grey images to GPU 
            enc_img = enc_img.cuda(local_rank, non_blocking=True)  # Move encrypted images to GPU 
            optimizer.zero_grad()  # Zero gradients 
            z_grey, z_enc = model(grey_img, enc_img)  # Forward pass through model 
            loss = criterion(z_grey, z_enc)  # Compute InfoNCE loss 
            loss.backward()  # Backpropagate loss 
            optimizer.step()  # Update model parameters 
            total_loss += loss.item()  # Accumulate loss 
            num_batches += 1  # Increment batch count 

        avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute average loss 
        if local_rank == 0:  # Only on main process 
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")  # Print epoch loss 
            save_path = os.path.join(  # Path to save model checkpoint 
                model_subdir,
                f"model_vit_clip_{vit_variant}_ddp_epoch_{epoch + 1}.pth"
            )
            torch.save(model.module.state_dict(), save_path)  # Save model weights 
            log_dict["epoch_logs"].append({  # Log epoch statistics 
                "epoch": epoch + 1,  # Current epoch 
                "loss": avg_loss,  # Average loss 
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp for epoch 
            })
            with open(log_path, "w", encoding="utf-8") as f:  # Open log file 
                json.dump(log_dict, f, ensure_ascii=False, indent=2)  # Write log to file 

    dist.destroy_process_group()  # Clean up distributed training resources 

def run_ddp(args):  # Launch distributed training 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # Set visible GPUs 
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Set master address for DDP 
    os.environ['MASTER_PORT'] = '29500'  # Set master port for DDP 
    world_size = torch.cuda.device_count()  # Number of GPUs to use 
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)  # Spawn processes for each GPU 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT-CLIP Dual Encoder DDP Training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--vit_variant', type=str, default='b_16', choices=['b_16', 'l_16'])
    args = parser.parse_args()
    run_ddp(args)
