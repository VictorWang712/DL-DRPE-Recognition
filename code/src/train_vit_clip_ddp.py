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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    vit_variant = args.vit_variant

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
        transforms.Resize((224, 224)),  # ViT默认224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    model = DualViTCLIP(
        embedding_dim=embedding_dim,
        vit_variant=vit_variant,
        pretrained=True
    ).cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = InfoNCELoss(temperature=0.07).cuda(local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        num_batches = 0
        for grey_img, enc_img, _ in tqdm(
            dataloader,
            desc=f"[GPU {local_rank}] Epoch {epoch + 1}/{num_epochs}",
            disable=(local_rank != 0)
        ):
            grey_img = grey_img.cuda(local_rank, non_blocking=True)
            enc_img = enc_img.cuda(local_rank, non_blocking=True)
            optimizer.zero_grad()
            z_grey, z_enc = model(grey_img, enc_img)
            loss = criterion(z_grey, z_enc)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if local_rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            save_path = os.path.join(
                model_subdir,
                f"model_vit_clip_{vit_variant}_ddp_epoch_{epoch + 1}.pth"
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
    parser = argparse.ArgumentParser(description='ViT-CLIP Dual Encoder DDP Training')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='CUDA_VISIBLE_DEVICES, e.g. "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--vit_variant', type=str, default='b_16', choices=['b_16', 'l_16'])
    args = parser.parse_args()
    run_ddp(args)
