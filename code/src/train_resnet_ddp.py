import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dataset import DRPESiameseDataset
from siamese_network_resnet import SiameseNetworkResNet
from triplet_loss import TripletLoss
import torchvision.transforms as transforms
import random
import numpy as np
import argparse
from datetime import datetime
import json

def set_seed(seed=42):  # Set random seeds for reproducibility 
    random.seed(seed)  # Set seed for Python random 
    np.random.seed(seed)  # Set seed for numpy random 
    torch.manual_seed(seed)  # Set seed for torch CPU 
    torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices 

class TripletDataset(Dataset):  # Custom dataset for generating triplets 
    def __init__(self, dataset, num_neg=5):  # Initialize with base dataset and number of negatives 
        self.dataset = dataset  # Store reference to base dataset 
        self.num_neg = num_neg  # Store number of negatives per anchor 
        self.triplets = self._make_triplets()  # Generate all triplets 

    def _make_triplets(self):  # Generate triplet indices for dataset 
        triplets = []  # List to store triplets 
        n = len(self.dataset)  # Get dataset size 
        for i in range(n):  # Iterate over all samples 
            anchor_idx = i  # Anchor index is current index 
            positive_idx = i  # Positive index is same as anchor 
            negs = set()  # Set to store unique negatives 
            while len(negs) < self.num_neg:  # Ensure enough negatives 
                neg_idx = random.randint(0, n - 1)  # Random negative index 
                if neg_idx != anchor_idx:  # Ensure negative is not anchor 
                    negs.add(neg_idx)  # Add negative index 
            for neg_idx in negs:  # For each negative index 
                triplets.append((anchor_idx, positive_idx, neg_idx))  # Append triplet 
        return triplets  # Return all triplets 

    def __len__(self):  # Return total number of triplets 
        return len(self.triplets)  # Length is number of triplets 

    def __getitem__(self, idx):  # Get triplet by index 
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]  # Unpack triplet indices 
        anchor_img, _, _ = self.dataset[anchor_idx]  # Get anchor image 
        _, positive_img, _ = self.dataset[positive_idx]  # Get positive image 
        _, negative_img, _ = self.dataset[negative_idx]  # Get negative image 
        return anchor_img, positive_img, negative_img  # Return images as a triplet 

def main_worker(local_rank, world_size, args):  # Main function for each process 
    set_seed()  # Set random seed for reproducibility 
    torch.cuda.set_device(local_rank)  # Set CUDA device for this process 
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)  # Initialize distributed process group 

    grey_root = '../../data/grey'  # Path to original images 
    encrypted_root = '../../data/drpe_encrypted'  # Path to encrypted images 
    batch_size = args.batch_size  # Batch size from arguments 
    num_epochs = args.epochs  # Number of epochs 
    lr = args.lr  # Learning rate 
    embedding_dim = args.embedding_dim  # Embedding dimension 

    base_model_dir = '../../model'  # Base directory for saving models 
    os.makedirs(base_model_dir, exist_ok=True)  # Create directory if not exists 
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Current timestamp 
    model_subdir = os.path.join(base_model_dir, f'run_{run_time}')  # Subdirectory for this run 
    os.makedirs(model_subdir, exist_ok=True)  # Create run subdirectory 

    log_dict = {  # Dictionary to store logs 
        "start_time": run_time,  # Log start time 
        "args": vars(args),  # Log arguments 
        "model_subdir": model_subdir,  # Log model directory 
        "epoch_logs": []  # List for epoch logs 
    }
    log_path = os.path.join(model_subdir, "train_log.json")  # Path for log file 

    transform = transforms.Compose([  # Compose image transformations 
        transforms.Resize((128, 128)),  # Resize images 
        transforms.ToTensor(),  # Convert to tensor 
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images 
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)  # Initialize custom dataset 
    triplet_dataset = TripletDataset(dataset, num_neg=5)  # Wrap with triplet dataset 
    sampler = DistributedSampler(triplet_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)  # Distributed sampler for DDP 
    dataloader = DataLoader(triplet_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)  # DataLoader with sampler 

    model = SiameseNetworkResNet(embedding_dim=embedding_dim).cuda(local_rank)  # Create model and move to device 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # Wrap model for DDP 
    criterion = TripletLoss(margin=1.0).cuda(local_rank)  # Triplet loss function 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # AdamW optimizer 

    for epoch in range(num_epochs):  # Loop over epochs 
        model.train()  # Set model to training mode 
        sampler.set_epoch(epoch)  # Shuffle data for each epoch 
        total_loss = 0  # Track total loss 
        num_batches = 0  # Track number of batches 
        for anchor_img, positive_img, negative_img in tqdm(  # Iterate over batches 
            dataloader,
            desc=f"[GPU {local_rank}] Epoch {epoch + 1}/{num_epochs}",  # Progress bar description 
            disable=(local_rank != 0)  # Only show progress on rank 0 
        ):
            anchor_img = anchor_img.cuda(local_rank, non_blocking=True)  # Move anchor to device 
            positive_img = positive_img.cuda(local_rank, non_blocking=True)  # Move positive to device 
            negative_img = negative_img.cuda(local_rank, non_blocking=True)  # Move negative to device 
            optimizer.zero_grad()  # Zero gradients 
            anchor_emb = model.module.forward_once(anchor_img)  # Forward anchor 
            positive_emb = model.module.forward_once(positive_img)  # Forward positive 
            negative_emb = model.module.forward_once(negative_img)  # Forward negative 
            loss = criterion(anchor_emb, positive_emb, negative_emb)  # Compute triplet loss 
            loss.backward()  # Backpropagate loss 
            optimizer.step()  # Update model parameters 
            total_loss += loss.item()  # Accumulate loss 
            num_batches += 1  # Increment batch count 

        avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute average loss 
        if local_rank == 0:  # Only save and log on main process 
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")  # Print epoch loss 
            save_path = os.path.join(  # Path to save model 
                model_subdir,
                f"model_resnet34_triplet_ddp_epoch_{epoch + 1}.pth"
            )
            torch.save(model.module.state_dict(), save_path)  # Save model weights 
            log_dict["epoch_logs"].append({  # Append epoch log 
                "epoch": epoch + 1,
                "loss": avg_loss,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            with open(log_path, "w", encoding="utf-8") as f:  # Write log file 
                json.dump(log_dict, f, ensure_ascii=False, indent=2)  # Dump log as JSON 

    dist.destroy_process_group()  # Clean up distributed process group 

def run_ddp(args):  # Function to run DDP training 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # Set visible GPUs 
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Set master address 
    os.environ['MASTER_PORT'] = '29500'  # Set master port 
    world_size = torch.cuda.device_count()  # Get number of GPUs 
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)  # Spawn processes for DDP 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet34 Siamese DDP Training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    args = parser.parse_args()
    run_ddp(args)
