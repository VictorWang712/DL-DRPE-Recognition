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
                neg_idx = random.randint(0, n-1)
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
    batch_size = 32
    num_epochs = 100
    lr = 1e-3
    embedding_dim = 256

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        num_batches = 0
        for anchor_img, positive_img, negative_img in tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}/{num_epochs}", disable=(local_rank != 0)):
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
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            torch.save(model.module.state_dict(), f'../../model/model_resnet_triplet_ddp_epoch_{epoch}.pth')

    dist.destroy_process_group()

def run_ddp():
    # 手动指定要用的GPU卡号
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, None), nprocs=world_size, join=True)


if __name__ == '__main__':
    run_ddp()
