# code/src/train_resnet_ddp.py

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TripletDataset(Dataset):
    def __init__(self, dataset, num_neg=5):
        self.dataset = dataset
        self.num_neg = num_neg
        self.triplets = self._make_triplets()

    def _make_triplets(self):
        triplets = []
        n = len(self.dataset)
        for i in range(n):
            anchor_idx = i
            positive_idx = i
            negs = set()
            while len(negs) < self.num_neg:
                neg_idx = random.randint(0, n - 1)
                if neg_idx != anchor_idx:
                    negs.add(neg_idx)
            for neg_idx in negs:
                triplets.append((anchor_idx, positive_idx, neg_idx))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor_img, _, _ = self.dataset[anchor_idx]
        _, positive_img, _ = self.dataset[positive_idx]
        _, negative_img, _ = self.dataset[negative_idx]
        return anchor_img, positive_img, negative_img

def main_worker(local_rank, world_size, args):
    set_seed()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)

    grey_root = '../../data/grey'
    encrypted_root = '../../data/drpe_encrypted'
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    embedding_dim = args.embedding_dim

    base_model_dir = '../../model'
    os.makedirs(base_model_dir, exist_ok=True)
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_subdir = os.path.join(base_model_dir, f'run_{run_time}')
    os.makedirs(model_subdir, exist_ok=True)

    log_dict = {
        "start_time": run_time,
        "args": vars(args),
        "model_subdir": model_subdir,
        "epoch_logs": []
    }
    log_path = os.path.join(model_subdir, "train_log.json")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)
    triplet_dataset = TripletDataset(dataset, num_neg=5)
    sampler = DistributedSampler(triplet_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(triplet_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    model = SiameseNetworkResNet(embedding_dim=embedding_dim).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = TripletLoss(margin=1.0).cuda(local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        num_batches = 0
        for anchor_img, positive_img, negative_img in tqdm(
            dataloader,
            desc=f"[GPU {local_rank}] Epoch {epoch + 1}/{num_epochs}",
            disable=(local_rank != 0)
        ):
            anchor_img = anchor_img.cuda(local_rank, non_blocking=True)
            positive_img = positive_img.cuda(local_rank, non_blocking=True)
            negative_img = negative_img.cuda(local_rank, non_blocking=True)
            optimizer.zero_grad()
            anchor_emb = model.module.forward_once(anchor_img)
            positive_emb = model.module.forward_once(positive_img)
            negative_emb = model.module.forward_once(negative_img)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if local_rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            save_path = os.path.join(
                model_subdir,
                f"model_resnet34_triplet_ddp_epoch_{epoch + 1}.pth"
            )
            torch.save(model.module.state_dict(), save_path)
            log_dict["epoch_logs"].append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_dict, f, ensure_ascii=False, indent=2)

    dist.destroy_process_group()

def run_ddp(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet34 Siamese DDP Training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    args = parser.parse_args()
    run_ddp(args)
