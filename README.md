# DL-DRPE-Recognition

A deep learning framework for matching original grayscale images and their encrypted versions using Double Random Phase Encoding (DRPE) and Siamese networks (ResNet, ConvNeXt, ViT-CLIP). Supports distributed multi-GPU training and retrieval with top-1/top-5 accuracy evaluation.

## Environment Setup

Install the required dependencies:

```bash
pip install torch torchvision tqdm pillow
```

## Data Preparation

### Extract Raw Images

Concatenate and extract the raw image dataset:

```bash
cd data
cat raw.tar.gz.part_* > raw.tar.gz
tar -xvzf raw.tar.gz
```

### Crop and Convert to Grayscale

Preprocess the raw images (crop, resize, grayscale):

```bash
cd code/src
python crop_and_gray_images.py
```

This will generate the `grey/` folder with preprocessed grayscale images.

## DRPE Image Encryption

Encrypt the grayscale images using Double Random Phase Encoding (DRPE):

```bash
python drpe_encrypt.py --src_root <path_to_grey_images> --dst_root <output_encrypted_dir> --phase1_seed <seed1> --phase2_seed <seed2>
```

Example:

```bash
python drpe_encrypt.py --src_root ../../data/grey --dst_root ../../data/drpe_encrypted --phase1_seed 42 --phase2_seed 123
```

## Model Training

This project supports three backbone architectures for Siamese retrieval: ResNet34, ConvNeXt, and ViT-CLIP.
All models use Triplet Loss and support distributed multi-GPU training (DDP).

### ResNet34 Siamese

General Command:

```bash
python train_resnet_ddp.py --gpus <gpu_ids> --batch_size <batch_size> --epochs <num_epochs> --embedding_dim <dim> --lr <learning_rate>
```

Parameters:

- `--gpus`: GPUs to use, e.g. "0,1,2,3"
- `--batch_size`: Batch size per process (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--embedding_dim`: Embedding dimension (default: 256)
- `--lr`: Learning rate (default: 1e-4)

Example:

```bash
python train_resnet_ddp.py --gpus 0,1,2,3 --batch_size 16 --epochs 50 --embedding_dim 256
```

### ConvNeXt Siamese

General Command:

```bash
python train_convnext_ddp.py --gpus <gpu_ids> --batch_size <batch_size> --epochs <num_epochs> --convnext_variant <variant> --embedding_dim <dim> --lr <learning_rate>
```

Parameters:

- `--gpus`: GPUs to use, e.g. "0,1,2,3"
- `--batch_size`: Batch size per process (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--convnext_variant`: ConvNeXt variant, one of `tiny`, `small`, `base`, `large` (default: `base`)
- `--embedding_dim`: Embedding dimension (default: 256)
- `--lr`: Learning rate (default: 1e-4)

Example:

```bash
python train_convnext_ddp.py --gpus 0,1,2,3 --batch_size 16 --epochs 50 --convnext_variant tiny
```

### ViT-CLIP Siamese

General Command:

```bash
python train_vit_clip_ddp.py --gpus <gpu_ids> --batch_size <batch_size> --epochs <num_epochs> --vit_variant <variant> --embedding_dim <dim> --lr <learning_rate>
```

Parameters:

- `--gpus`: GPUs to use, e.g. "0,1,2,3"
- `--batch_size`: Batch size per process (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--vit_variant`: ViT variant, one of `b_16`, `l_16` (default: `b_16`)
- `--embedding_dim`: Embedding dimension (default: 256)
- `--lr`: Learning rate (default: 1e-4)

Example:

```bash
python train_vit_clip_ddp.py --gpus 0,1,2,3 --batch_size 8 --epochs 50 --vit_variant b_16
```

## Image Retrieval & Evaluation

Evaluate the trained model by retrieving the original image for each encrypted image, and compute top-1/top-5 accuracy.

Retrieval results (including per-class statistics and detailed matches) are saved as a JSON file in the `../../result/` directory.

### ResNet34 Siamese Retrieval

General Command:

```bash
python retrieve_resnet.py --model_path <model_checkpoint_path> --embedding_dim <dim> [--cpu]
```

Parameters:

- `--model_path`: Path to the trained model checkpoint
- `--embedding_dim`: Embedding dimension used during training (default: 256)
- `--cpu`: (Optional) Force inference on CPU

Example:

```bash
python retrieve_resnet.py --model_path ../../model/run_19700101_000000/model_resnet34_triplet_ddp_epoch_50.pth --embedding_dim 256
```

### ConvNeXt Siamese Retrieval

General Command:

```bash
python retrieve_convnext.py --model_path <model_checkpoint_path> --convnext_variant <variant> [--cpu]
```

Parameters:

- `--model_path`: Path to the trained model checkpoint
- `--convnext_variant`: ConvNeXt variant used during training (`tiny`, `small`, `base`, `large`)
- `--cpu`: (Optional) Force inference on CPU

Example:

```bash
python retrieve_convnext.py --model_path ../../model/run_19700101_000000/model_convnext_base_triplet_ddp_epoch_50.pth --convnext_variant tiny
```

### ViT-CLIP Siamese Retrieval

General Command:

```bash
python retrieve_vit_clip.py --model_path <model_checkpoint_path> --vit_variant <variant> --embedding_dim <dim> [--cpu]
```

Parameters:

- `--model_path`: Path to the trained model checkpoint
- `--vit_variant`: ViT variant used during training, one of `b_16`, `l_16` (default: `b_16`)
- `--embedding_dim`: Embedding dimension used during training (default: 256)
- `--cpu`: (Optional) Force inference on CPU

Example:

```bash
python retrieve_vit_clip.py --model_path ../../model/run_19700101_000000/model_vit_clip_b_16_triplet_ddp_epoch_50.pth --vit_variant b_16 --embedding_dim 256
```

## Results

The retrieval script reports:

- Top-1 Accuracy: Percentage of encrypted images whose closest match is the correct original image.
- Top-5 Accuracy: Percentage where the correct original is among the 5 closest matches.
- Per-class statistics: Accuracy for each class/category.
- Detailed results: For each encrypted image, the top-5 matches and correctness.
