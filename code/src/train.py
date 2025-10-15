import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 只用0,1,2,3号卡

import torch
from torch.utils.data import DataLoader, Dataset
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

def main():
    set_seed()
    grey_root = '../../data/grey'
    encrypted_root = '../../data/drpe_encrypted'
    batch_size = 32
    num_epochs = 100
    lr = 1e-3
    embedding_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)
    triplet_dataset = TripletDataset(dataset, num_neg=5)
    dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SiameseNetworkResNet(embedding_dim=embedding_dim).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for DataParallel!")
        model = torch.nn.DataParallel(model)
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for anchor_img, positive_img, negative_img in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            optimizer.zero_grad()
            anchor_emb = model.module.forward_once(anchor_img) if hasattr(model, 'module') else model.forward_once(anchor_img)
            positive_emb = model.module.forward_once(positive_img) if hasattr(model, 'module') else model.forward_once(positive_img)
            negative_emb = model.module.forward_once(negative_img) if hasattr(model, 'module') else model.forward_once(negative_img)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f'../../model/model_resnet_triplet_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
