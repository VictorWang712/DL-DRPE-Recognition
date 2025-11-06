import os
import torch
from dataset import DRPESiameseDataset
from siamese_network_resnet import SiameseNetworkResNet
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime
import argparse

def main(args):
    grey_root = '../../data/grey'  # Path to the directory containing grey images 
    encrypted_root = '../../data/drpe_encrypted'  # Path to the directory containing encrypted images 
    model_path = args.model_path  # Path to the trained model checkpoint 
    embedding_dim = args.embedding_dim  # Dimensionality of the embedding vector 
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')  # Select device based on availability and user preference 

    result_dir = '../../result'  # Directory to save results 
    os.makedirs(result_dir, exist_ok=True)  # Create result directory if it doesn't exist 
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')  # Current timestamp for result file naming 
    result_json = os.path.join(result_dir, f'retrieve_resnet34_{now_str}.json')  # Path to save the result JSON 

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128 
        transforms.ToTensor(),  # Convert images to tensor 
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize tensor values 
    ])
    dataset = DRPESiameseDataset(grey_root, encrypted_root, transform)  # Initialize the dataset with transforms 

    model = SiameseNetworkResNet(embedding_dim=embedding_dim).to(device)  # Instantiate the model and move to device 
    print(f"Loading model: {model_path}")  # Print model loading message 
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights 
    model.eval()  # Set model to evaluation mode 

    grey_embeddings = []  # List to store embeddings of gallery images 
    grey_names = []  # List to store names of gallery images 
    grey_classes = []  # List to store class labels of gallery images 
    with torch.no_grad():  # Disable gradient computation for efficiency 
        for i in tqdm(range(len(dataset)), desc="Building gallery"):  # Iterate over all images to build gallery 
            grey_img, _, name = dataset[i]  # Get grey image and its name 
            grey_img = grey_img.unsqueeze(0).to(device)  # Add batch dimension and move to device 
            emb = model.forward_once(grey_img)  # Compute embedding for the image 
            grey_embeddings.append(emb.cpu())  # Store embedding on CPU 
            folder = get_class_from_path(dataset.pairs[i][0])  # Extract class label from file path 
            img_name = os.path.basename(dataset.pairs[i][0])  # Extract image file name 
            full_name = f"{folder}/{img_name}"  # Combine class and file name 
            grey_names.append(full_name)  # Store full image name 
            grey_classes.append(folder)  # Store class label 
    grey_embeddings = torch.cat(grey_embeddings, dim=0)  # (N, D) Concatenate all embeddings into a tensor 

    top1_correct = 0  # Counter for top-1 correct retrievals 
    top5_correct = 0  # Counter for top-5 correct retrievals 
    total = 0  # Total number of queries 

    class_top1 = defaultdict(lambda: [0, 0])  # Per-class top-1 statistics 
    class_top5 = defaultdict(lambda: [0, 0])  # Per-class top-5 statistics 
    results = []  # List to store detailed retrieval results 

    for idx in tqdm(range(len(dataset)), desc="Retrieving"):  # Iterate over all queries 
        _, enc_img, enc_name = dataset[idx]  # Get encrypted image and its name 
        enc_img = enc_img.unsqueeze(0).to(device)  # Add batch dimension and move to device 
        with torch.no_grad():  # Disable gradient computation 
            enc_emb = model.forward_once(enc_img)  # Compute embedding for encrypted image 
            dists = torch.norm(grey_embeddings - enc_emb.cpu(), dim=1)  # Compute L2 distances to all gallery embeddings 
            sorted_indices = torch.argsort(dists)  # Sort indices by distance (ascending) 
            top5_indices = sorted_indices[:5]  # Indices of top-5 closest gallery images 
            top5_names = [grey_names[i] for i in top5_indices]  # Names of top-5 matches 
            match_name = grey_names[sorted_indices[0]]  # Name of top-1 match 

            folder = get_class_from_path(dataset.pairs[idx][0])  # Extract class label for current query 
            img_name = os.path.basename(dataset.pairs[idx][0])  # Extract image file name for query 
            true_name = f"{folder}/{img_name}"  # True name of the query image 
            true_class = folder  # True class label of the query 

            is_top1 = (match_name == true_name)  # Check if top-1 match is correct 
            is_top5 = (true_name in top5_names)  # Check if true image is in top-5 matches 
            top1_correct += int(is_top1)  # Update top-1 correct counter 
            top5_correct += int(is_top5)  # Update top-5 correct counter 
            total += 1  # Increment total queries 

            class_top1[true_class][0] += int(is_top1)  # Update per-class top-1 correct 
            class_top1[true_class][1] += 1  # Update per-class total 
            class_top5[true_class][0] += int(is_top5)  # Update per-class top-5 correct 
            class_top5[true_class][1] += 1  # Update per-class total 

            results.append({
                'EncryptedImage': enc_name,  # Name of the encrypted image 
                'Class': true_class,  # Class label of the image 
                'TrueName': true_name,  # Full name of the true image 
                'Top1Match': match_name,  # Name of the top-1 matched image 
                'IsTop1Correct': is_top1,  # Whether top-1 match is correct 
                'Top5Matches': top5_names,  # Names of top-5 matched images 
                'IsTop5Correct': is_top5  # Whether top-5 contains the correct image 
            })

    top1_acc = top1_correct / total if total > 0 else 0  # Calculate overall top-1 accuracy 
    top5_acc = top5_correct / total if total > 0 else 0  # Calculate overall top-5 accuracy 

    per_class_stats = {}  # Dictionary to store per-class statistics 
    for cls in sorted(class_top1.keys()):  # Iterate over all classes 
        c_total = class_top1[cls][1]  # Total queries for this class 
        c_top1_acc = class_top1[cls][0] / c_total if c_total > 0 else 0  # Top-1 accuracy for this class 
        c_top5_acc = class_top5[cls][0] / c_total if c_total > 0 else 0  # Top-5 accuracy for this class 
        per_class_stats[cls] = {
            'Top1Accuracy': c_top1_acc,  # Store top-1 accuracy 
            'Top5Accuracy': c_top5_acc,  # Store top-5 accuracy 
            'Total': c_total  # Store total number of queries 
        }

    summary = {
        'Top1Accuracy': top1_acc,  # Overall top-1 accuracy 
        'Top5Accuracy': top5_acc,  # Overall top-5 accuracy 
        'PerClass': per_class_stats,  # Per-class accuracy statistics 
        'Total': total,  # Total number of queries 
        'ModelInfo': {
            'ModelPath': os.path.abspath(model_path),  # Absolute path to model 
            'EmbeddingDim': embedding_dim,  # Embedding dimension used 
            'RetrieveTime': now_str  # Timestamp of retrieval 
        }
    }

    with open(result_json, 'w', encoding='utf-8') as f:  # Open result file for writing 
        json.dump({
            'Summary': summary,  # Write summary statistics 
            'Details': results  # Write detailed retrieval results 
        }, f, ensure_ascii=False, indent=2)  # Use pretty-printing for JSON 

    print(f"\nTop-1 Accuracy: {top1_acc:.4f}")  # Print overall top-1 accuracy 
    print(f"Top-5 Accuracy: {top5_acc:.4f}")  # Print overall top-5 accuracy 
    print(f"Detailed results saved to: {result_json}")  # Print path to result file 
    print(f"Model used: {os.path.abspath(model_path)}")  # Print absolute model path 

def get_class_from_path(path):
    return os.path.basename(os.path.dirname(path))  # Extract class label from file path 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet34 Siamese Retrieval')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()
    main(args)
