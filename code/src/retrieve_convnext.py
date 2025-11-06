import os
import torch
from dataset import DRPESiameseDataset
from siamese_network_convnext import SiameseNetworkConvNeXt
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime
import argparse

def main(args):
    grey_root = '../../data/grey'  # Path to grayscale images directory 
    encrypted_root = '../../data/drpe_encrypted'  # Path to encrypted images directory 

    model_path = args.model_path  # Model checkpoint path from command line 
    convnext_variant = args.convnext_variant  # ConvNeXt model variant from command line 
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')  # Select device 

    result_dir = '../../result'  # Directory to save results 
    os.makedirs(result_dir, exist_ok=True)  # Create result directory if it doesn't exist 
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')  # Current timestamp string 
    result_json = os.path.join(result_dir, f'retrieve_convnext_{now_str}.json')  # Output JSON path 

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128 
        transforms.ToTensor(),  # Convert images to tensor 
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)  # Initialize dataset 

    embedding_dim = 256  # Embedding vector dimension 
    model = SiameseNetworkConvNeXt(
        embedding_dim=embedding_dim,  # Set embedding dimension 
        convnext_variant=convnext_variant,  # Set ConvNeXt variant 
        pretrained=False  # Do not use pretrained weights 
    ).to(device)  # Move model to device 

    print(f"Loading model: {model_path}")  # Print model loading info 
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights 
    model.eval()  # Set model to evaluation mode 

    grey_embeddings = []  # List to store gallery embeddings 
    grey_names = []  # List to store gallery image names 
    grey_classes = []  # List to store gallery image classes 
    with torch.no_grad():  # Disable gradient computation 
        for i in tqdm(range(len(dataset)), desc="Building gallery"):  # Iterate over dataset for gallery 
            grey_img, _, name = dataset[i]  # Get grayscale image and its name 
            grey_img = grey_img.unsqueeze(0).to(device)  # Add batch dimension and move to device 
            emb = model.forward_once(grey_img)  # Get embedding from model 
            grey_embeddings.append(emb.cpu())  # Store embedding on CPU 
            grey_names.append(name)  # Store image name 
            grey_classes.append(get_class_from_path(dataset.pairs[i][0]))  # Store class label 
    grey_embeddings = torch.cat(grey_embeddings, dim=0)  # Concatenate all embeddings into one tensor 

    top1_correct = 0  # Counter for top-1 correct retrievals 
    top5_correct = 0  # Counter for top-5 correct retrievals 
    total = 0  # Total number of queries 

    class_top1 = defaultdict(lambda: [0, 0])  # Per-class top-1 stats 
    class_top5 = defaultdict(lambda: [0, 0])  # Per-class top-5 stats 
    results = []  # List to store detailed retrieval results 

    for idx in tqdm(range(len(dataset)), desc="Retrieving"):  # Iterate over queries 
        _, enc_img, enc_name = dataset[idx]  # Get encrypted image and its name 
        enc_img = enc_img.unsqueeze(0).to(device)  # Add batch dimension and move to device 
        with torch.no_grad():  # Disable gradient computation 
            enc_emb = model.forward_once(enc_img)  # Get embedding for encrypted image 
            dists = torch.norm(grey_embeddings - enc_emb.cpu(), dim=1)  # Compute L2 distances to gallery 
            sorted_indices = torch.argsort(dists)  # Sort indices by distance 
            top5_indices = sorted_indices[:5]  # Get indices of top-5 closest 
            top5_names = [grey_names[i] for i in top5_indices]  # Get names of top-5 matches 
            match_name = grey_names[sorted_indices[0]]  # Name of closest match 

            if enc_name.endswith('_mag.png'):  # Check if encrypted name ends with '_mag.png' 
                true_name = enc_name.replace('_mag.png', '.jpg')  # Recover original name 
            else:
                true_name = enc_name  # Use name as is 

            true_class = get_class_from_path(dataset.pairs[idx][0])  # Get class label for query 

            is_top1 = (match_name == true_name)  # Check if top-1 match is correct 
            is_top5 = (true_name in top5_names)  # Check if true name is in top-5 
            top1_correct += int(is_top1)  # Increment top-1 counter 
            top5_correct += int(is_top5)  # Increment top-5 counter 
            total += 1  # Increment total queries 

            class_top1[true_class][0] += int(is_top1)  # Update per-class top-1 
            class_top1[true_class][1] += 1  # Update per-class total 
            class_top5[true_class][0] += int(is_top5)  # Update per-class top-5 
            class_top5[true_class][1] += 1  # Update per-class total 

            results.append({
                'EncryptedImage': enc_name,  # Store encrypted image name 
                'Class': true_class,  # Store class label 
                'TrueName': true_name,  # Store ground truth name 
                'Top1Match': match_name,  # Store top-1 match name 
                'IsTop1Correct': is_top1,  # Store if top-1 is correct 
                'Top5Matches': top5_names,  # Store top-5 match names 
                'IsTop5Correct': is_top5  # Store if top-5 is correct 
            })

    top1_acc = top1_correct / total if total > 0 else 0  # Compute overall top-1 accuracy 
    top5_acc = top5_correct / total if total > 0 else 0  # Compute overall top-5 accuracy 

    per_class_stats = {}  # Dictionary for per-class stats 
    for cls in sorted(class_top1.keys()):  # Iterate over all classes 
        c_total = class_top1[cls][1]  # Total samples for class 
        c_top1_acc = class_top1[cls][0] / c_total if c_total > 0 else 0  # Per-class top-1 accuracy 
        c_top5_acc = class_top5[cls][0] / c_total if c_total > 0 else 0  # Per-class top-5 accuracy 
        per_class_stats[cls] = {
            'Top1Accuracy': c_top1_acc,  # Store per-class top-1 accuracy 
            'Top5Accuracy': c_top5_acc,  # Store per-class top-5 accuracy 
            'Total': c_total  # Store per-class total 
        }

    summary = {
        'Top1Accuracy': top1_acc,  # Overall top-1 accuracy 
        'Top5Accuracy': top5_acc,  # Overall top-5 accuracy 
        'PerClass': per_class_stats,  # Per-class statistics 
        'Total': total  # Total number of queries 
    }

    with open(result_json, 'w', encoding='utf-8') as f:  # Open result file for writing 
        json.dump({
            'Summary': summary,  # Save summary statistics 
            'Details': results  # Save detailed results 
        }, f, ensure_ascii=False, indent=2)  # Write as formatted JSON 

    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")  # Print top-1 accuracy 
    print(f"Top-5 Accuracy: {top5_acc:.4f}")  # Print top-5 accuracy 
    print(f"Detailed results saved to: {result_json}")  # Print result file path 
    print(f"Model used: {os.path.abspath(model_path)}")  # Print absolute model path 

def get_class_from_path(path):
    return os.path.basename(os.path.dirname(path))  # Extract class name from file path 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ConvNeXt Siamese Retrieval')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--convnext_variant', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()
    main(args)
