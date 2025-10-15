import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 只用0,1,2,3号卡

import torch
from dataset import DRPESiameseDataset
from siamese_network_resnet import SiameseNetworkResNet
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import json
from collections import defaultdict
from datetime import datetime

def main():
    grey_root = '../../data/grey'
    encrypted_root = '../../data/drpe_encrypted'
    model_path = '../../model/model_resnet_triplet_epoch_99.pth'  # 用你训练好的Triplet模型
    embedding_dim = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_dir = '../../result'
    os.makedirs(result_dir, exist_ok=True)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_json = os.path.join(result_dir, f'retrieve_result_{now_str}.json')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)

    # 构建所有原图的embedding库
    model = SiameseNetworkResNet(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    grey_embeddings = []
    grey_names = []
    grey_classes = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Building gallery"):
            grey_img, _, name = dataset[i]
            grey_img = grey_img.unsqueeze(0).to(device)
            emb = model.forward_once(grey_img)
            grey_embeddings.append(emb.cpu())
            grey_names.append(name)
            grey_classes.append(get_class_from_path(dataset.pairs[i][0]))
    grey_embeddings = torch.cat(grey_embeddings, dim=0)  # (N, D)

    # 检索测试
    top1_correct = 0
    top5_correct = 0
    total = 0

    # per-class统计
    class_top1 = defaultdict(lambda: [0, 0])  # class_name: [correct, total]
    class_top5 = defaultdict(lambda: [0, 0])

    # 保存详细结果
    results = []
    for idx in tqdm(range(len(dataset)), desc="Retrieving"):
        _, enc_img, enc_name = dataset[idx]
        enc_img = enc_img.unsqueeze(0).to(device)
        with torch.no_grad():
            enc_emb = model.forward_once(enc_img)
            dists = torch.norm(grey_embeddings - enc_emb.cpu(), dim=1)
            sorted_indices = torch.argsort(dists)  # 从小到大排序
            top5_indices = sorted_indices[:5]
            top5_names = [grey_names[i] for i in top5_indices]
            match_name = grey_names[sorted_indices[0]]

            # 获取原图真实名字
            if enc_name.endswith('_mag.png'):
                true_name = enc_name.replace('_mag.png', '.jpg')
            else:
                true_name = enc_name

            # 获取类别
            true_class = get_class_from_path(dataset.pairs[idx][0])

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

    summary = {
        'Top1Accuracy': top1_acc,
        'Top5Accuracy': top5_acc,
        'PerClass': per_class_stats,
        'Total': total
    }

    # 保存JSON
    with open(result_json, 'w', encoding='utf-8') as f:
        json.dump({
            'Summary': summary,
            'Details': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Per-class统计与详细检索结果已保存至: {result_json}")

def get_class_from_path(path):
    # 获取类别名（假设类别为上一级文件夹名）
    return os.path.basename(os.path.dirname(path))

if __name__ == '__main__':
    main()
