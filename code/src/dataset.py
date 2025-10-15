import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DRPESiameseDataset(Dataset):
    def __init__(self, grey_root, encrypted_root, transform=None):
        self.grey_root = grey_root
        self.encrypted_root = encrypted_root
        self.transform = transform if transform else transforms.ToTensor()
        self.pairs = self._make_pairs()

    def _make_pairs(self):
        pairs = []
        for cls in os.listdir(self.grey_root):
            grey_cls_dir = os.path.join(self.grey_root, cls)
            enc_cls_dir = os.path.join(self.encrypted_root, cls)
            if not os.path.isdir(grey_cls_dir):
                continue
            for fname in os.listdir(grey_cls_dir):
                if not fname.endswith('.jpg'):
                    continue
                grey_path = os.path.join(grey_cls_dir, fname)
                enc_fname = fname + '_mag.png' if not fname.endswith('_mag.png') else fname
                enc_fname = fname.replace('.jpg', '.jpg_mag.png')
                enc_path = os.path.join(enc_cls_dir, enc_fname)
                if os.path.exists(enc_path):
                    pairs.append((grey_path, enc_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        grey_path, enc_path = self.pairs[idx]
        grey_img = Image.open(grey_path).convert('L')
        enc_img = Image.open(enc_path).convert('L')
        grey_img = self.transform(grey_img)
        enc_img = self.transform(enc_img)
        return grey_img, enc_img, os.path.basename(grey_path)
