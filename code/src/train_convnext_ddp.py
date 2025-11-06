import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from dataset import DRPESiameseDataset
from siamese_network_convnext import SiameseNetworkConvNeXt
from triplet_loss import TripletLoss
import torchvision.transforms as transforms
import random
import numpy as np
import argparse
from datetime import datetime
import json

def set_seed(seed=42):  # Set random seeds for reproducibility 
    random.seed(seed)  # Set Python random seed 
    np.random.seed(seed)  # Set numpy random seed 
    torch.manual_seed(seed)  # Set torch CPU seed 
    torch.cuda.manual_seed_all(seed)  # Set torch CUDA seed for all GPUs 

class TripletDataset(Dataset):  # Custom dataset for triplet sampling 
    def __init__(self, dataset, num_neg=5):  # Initialize with base dataset and number of negatives 
        self.dataset = dataset  # Store reference to base dataset 
        self.num_neg = num_neg  # Number of negative samples per anchor 
        self.triplets = self._make_triplets()  # Precompute all triplets 

    def _make_triplets(self):  # Generate triplets for training 
        triplets = []  # List to store triplet indices 
        n = len(self.dataset)  # Total number of samples 
        for i in range(n):  # Iterate over all samples as anchor 
            anchor_idx = i  # Anchor index 
            positive_idx = i  # Positive is same as anchor 
            negs = set()  # Set to store unique negative indices 
            while len(negs) < self.num_neg:  # Sample until enough negatives 
                neg_idx = random.randint(0, n - 1)  # Random negative index 
                if neg_idx != anchor_idx:  # Ensure negative is not anchor 
                    negs.add(neg_idx)  # Add negative index 
            for neg_idx in negs:  # For each negative 
                triplets.append((anchor_idx, positive_idx, neg_idx))  # Store the triplet 
        return triplets  # Return all triplets 

    def __len__(self):  # Return number of triplets 
        return len(self.triplets)  # Number of triplets 

    def __getitem__(self, idx):  # Get a triplet by index 
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]  # Unpack triplet indices 
        anchor_img, _, _ = self.dataset[anchor_idx]  # Get anchor image 
        _, positive_img, _ = self.dataset[positive_idx]  # Get positive image 
        _, negative_img, _ = self.dataset[negative_idx]  # Get negative image 
        return anchor_img, positive_img, negative_img  # Return triplet images 

def main_worker(local_rank, world_size, args):  # Main function for each DDP process 
    set_seed()  # Set random seeds 
    torch.cuda.set_device(local_rank)  # Set CUDA device for this process 
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)  # Initialize DDP 

    grey_root = '../../data/grey'  # Path to original images 
    encrypted_root = '../../data/drpe_encrypted'  # Path to encrypted images 
    batch_size = args.batch_size  # Batch size from arguments 
    num_epochs = args.epochs  # Number of epochs from arguments 
    lr = args.lr  # Learning rate from arguments 
    embedding_dim = args.embedding_dim  # Embedding dimension from arguments 
    convnext_variant = args.convnext_variant  # ConvNeXt variant from arguments 

    base_model_dir = '../../model'  # Directory to save models 
    os.makedirs(base_model_dir, exist_ok=True)  # Ensure model directory exists 
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Current timestamp 
    model_subdir = os.path.join(base_model_dir, f'run_{run_time}')  # Subdirectory for this run 
    os.makedirs(model_subdir, exist_ok=True)  # Create run directory 

    log_dict = {  # Dictionary to store training logs 
        "start_time": run_time,  # Training start time 
        "args": vars(args),  # Training arguments 
        "model_subdir": model_subdir,  # Model directory 
        "epoch_logs": []  # List for per-epoch logs 
    }
    log_path = os.path.join(model_subdir, "train_log.json")  # Path for log file 

    transform = transforms.Compose([  # Image preprocessing pipeline 
        transforms.Resize((128, 128)),  # Resize images 
        transforms.ToTensor(),  # Convert to tensor 
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize 
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)  # Load custom dataset 
    triplet_dataset = TripletDataset(dataset, num_neg=5)  # Wrap with triplet dataset 
    sampler = DistributedSampler(triplet_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)  # Distributed sampler 
    dataloader = DataLoader(triplet_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)  # DataLoader for triplets 

    model = SiameseNetworkConvNeXt(  # Instantiate Siamese ConvNeXt model 
        embedding_dim=embedding_dim,  # Set embedding dimension 
        convnext_variant=convnext_variant,  # Set ConvNeXt variant 
        pretrained=True  # Use pretrained weights 
    ).cuda(local_rank)  # Move model to GPU 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # Wrap model with DDP 
    criterion = TripletLoss(margin=1.0).cuda(local_rank)  # Triplet loss function on GPU 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)  # AdamW optimizer 

    for epoch in range(num_epochs):  # Training loop over epochs 
        model.train()  # Set model to training mode 
        sampler.set_epoch(epoch)  # Shuffle data for each epoch 
        total_loss = 0  # Accumulate total loss 
        num_batches = 0  # Count number of batches 
        for anchor_img, positive_img, negative_img in tqdm(  # Iterate over DataLoader 
            dataloader,
            desc=f"[GPU {local_rank}] Epoch {epoch + 1}/{num_epochs}",  # Progress bar description 
            disable=(local_rank != 0)  # Only show progress bar on rank 0 
        ):
            anchor_img = anchor_img.cuda(local_rank, non_blocking=True)  # Move anchor to GPU 
            positive_img = positive_img.cuda(local_rank, non_blocking=True)  # Move positive to GPU 
            negative_img = negative_img.cuda(local_rank, non_blocking=True)  # Move negative to GPU 
            optimizer.zero_grad()  # Clear gradients 
            anchor_emb = model.module.forward_once(anchor_img)  # Forward pass for anchor 
            positive_emb = model.module.forward_once(positive_img)  # Forward pass for positive 
            negative_emb = model.module.forward_once(negative_img)  # Forward pass for negative 
            loss = criterion(anchor_emb, positive_emb, negative_emb)  # Compute triplet loss 
            loss.backward()  # Backpropagate loss 
            optimizer.step()  # Update model parameters 
            total_loss += loss.item()  # Accumulate loss 
            num_batches += 1  # Increment batch count 

        avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute average loss 
        if local_rank == 0:  # Only on main process 
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")  # Print loss 
            save_path = os.path.join(  # Path to save model 
                model_subdir,
                f"model_convnext_{convnext_variant}_triplet_ddp_epoch_{epoch + 1}.pth"
            )
            torch.save(model.module.state_dict(), save_path)  # Save model checkpoint 
            log_dict["epoch_logs"].append({  # Add epoch log 
                "epoch": epoch + 1,  # Current epoch 
                "loss": avg_loss,  # Average loss 
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp 
            })
            with open(log_path, "w", encoding="utf-8") as f:  # Open log file 
                json.dump(log_dict, f, ensure_ascii=False, indent=2)  # Write logs to file 

    dist.destroy_process_group()  # Clean up distributed training 

def run_ddp(args):  # Launch DDP processes 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # Set visible GPUs 
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Set master address 
    os.environ['MASTER_PORT'] = '29500'  # Set master port 
    world_size = torch.cuda.device_count()  # Number of GPUs 
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)  # Spawn processes 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConvNeXt Siamese DDP Training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--convnext_variant', type=str, default='base', choices=['tiny', 'small', 'base', 'large'])
    args = parser.parse_args()
    run_ddp(args)
