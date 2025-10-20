import os
import torch
from dataset import DRPESiameseDataset
from siamese_network_vit_clip import DualViTCLIP
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime
import argparse

def main(args):
    grey_root = '../../data/grey'
    encrypted_root = '../../data/drpe_encrypted'
    model_path = args.model_path
    vit_variant = args.vit_variant
    embedding_dim = args.embedding_dim
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    result_dir = '../../result'
    os.makedirs(result_dir, exist_ok=True)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_json = os.path.join(result_dir, f'retrieve_vit_clip_{now_str}.json')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)

    model = DualViTCLIP(
        embedding_dim=embedding_dim,
        vit_variant=vit_variant,
        pretrained=False
    ).to(device)

    print(f"Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    grey_embeddings = []
    grey_names = []
    grey_classes = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Building gallery"):
            grey_img, _, name = dataset[i]
            grey_img = grey_img.unsqueeze(0).to(device)
            emb = model.encode_grey(grey_img)
            grey_embeddings.append(emb.cpu())
            # 获取类别和图片名，拼成 folder/image_xxxx.jpg
            folder = get_class_from_path(dataset.pairs[i][0])
            img_name = os.path.basename(dataset.pairs[i][0])
            full_name = f"{folder}/{img_name}"
            grey_names.append(full_name)
            grey_classes.append(folder)
    grey_embeddings = torch.cat(grey_embeddings, dim=0)  # (N, D)

    top1_correct = 0
    top5_correct = 0
    total = 0

    class_top1 = defaultdict(lambda: [0, 0])
    class_top5 = defaultdict(lambda: [0, 0])
    results = []

    for idx in tqdm(range(len(dataset)), desc="Retrieving"):
        _, enc_img, enc_name = dataset[idx]
        enc_img = enc_img.unsqueeze(0).to(device)
        with torch.no_grad():
            enc_emb = model.encode_enc(enc_img)
            dists = torch.norm(grey_embeddings - enc_emb.cpu(), dim=1)
            sorted_indices = torch.argsort(dists)
            top5_indices = sorted_indices[:5]
            top5_names = [grey_names[i] for i in top5_indices]
            match_name = grey_names[sorted_indices[0]]

            # 获取当前加密图片的类别和图片名，拼成 folder/image_xxxx.jpg
            folder = get_class_from_path(dataset.pairs[idx][0])
            img_name = os.path.basename(dataset.pairs[idx][0])
            true_name = f"{folder}/{img_name}"

            true_class = folder

            is_top1 = (match_name == true_name)
            is_top5 = (true_name in top5_names)
            top1_correct += int(is_top1)
            top5_correct += int(is_top5)
            total += 1

            class_top1[true_class][0] += int(is_top1)
            class_top1[true_class][1] += 1
            class_top5[true_class][0] += int(is_top5)
            class_top5[true_class][1] += 1

            results.append({
                'EncryptedImage': enc_name,
                'Class': true_class,
                'TrueName': true_name,
                'Top1Match': match_name,
                'IsTop1Correct': is_top1,
                'Top5Matches': top5_names,
                'IsTop5Correct': is_top5
            })

    top1_acc = top1_correct / total if total > 0 else 0
    top5_acc = top5_correct / total if total > 0 else 0

    per_class_stats = {}
    for cls in sorted(class_top1.keys()):
        c_total = class_top1[cls][1]
        c_top1_acc = class_top1[cls][0] / c_total if c_total > 0 else 0
        c_top5_acc = class_top5[cls][0] / c_total if c_total > 0 else 0
        per_class_stats[cls] = {
            'Top1Accuracy': c_top1_acc,
            'Top5Accuracy': c_top5_acc,
            'Total': c_total
        }

    # 记录检索参数和模型信息
    summary = {
        'Top1Accuracy': top1_acc,
        'Top5Accuracy': top5_acc,
        'PerClass': per_class_stats,
        'Total': total,
        'ModelInfo': {
            'ModelPath': os.path.abspath(model_path),
            'ViTVariant': vit_variant,
            'EmbeddingDim': embedding_dim,
            'RetrieveTime': now_str
        }
    }

    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump({
            'Summary': summary,
            'Details': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Detailed results saved to: {result_json}")
    print(f"Model used: {os.path.abspath(model_path)}")

def get_class_from_path(path):
    return os.path.basename(os.path.dirname(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT-CLIP Dual Encoder Retrieval')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vit_variant', type=str, default='b_16', choices=['b_16', 'l_16'])
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()
    main(args)
