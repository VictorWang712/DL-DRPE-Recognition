# DL-DRPE-Recognition

A deep learning framework for matching original grayscale images and their encrypted versions using Double Random Phase Encoding (DRPE) and Siamese networks (ResNet/ConvNeXt backbone, Triplet Loss).  
Supports distributed multi-GPU training and retrieval with top-1/top-5 accuracy evaluation.

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

Train a Siamese network with a ConvNeXt backbone using distributed data parallel (DDP) and Triplet Loss.

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

## Image Retrieval & Evaluation

Evaluate the trained model by retrieving the original image for each encrypted image, and compute top-1/top-5 accuracy.

General Command:

```bash
python retrieve.py --model_path <model_checkpoint_path> --convnext_variant <variant> [--cpu]
```

Parameters:

- `--model_path`: Path to the trained model checkpoint (e.g., `../../model/run_xxx/model_convnext_tiny_triplet_ddp_epoch_50.pth`)
- `--convnext_variant`: ConvNeXt variant used during training (`tiny`, `small`, `base`, `large`)
- `--cpu`: (Optional) Force inference on CPU

Example:

```bash
python retrieve.py --model_path ../../model/run_20251016_000000/model_convnext_base_triplet_ddp_epoch_50.pth --convnext_variant base
```

Retrieval results (including per-class statistics and detailed matches) are saved as a JSON file in the `../../result/` directory.

## Results

The retrieval script reports:

- Top-1 Accuracy: Percentage of encrypted images whose closest match is the correct original image.
- Top-5 Accuracy: Percentage where the correct original is among the 5 closest matches.
- Per-class statistics: Accuracy for each class/category.
- Detailed results: For each encrypted image, the top-5 matches and correctness.

Results are saved in a JSON file, e.g., `../../result/retrieve_result_20251016_000000.json`.
