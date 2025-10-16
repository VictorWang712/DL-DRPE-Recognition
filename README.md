# DL-DRPE-Recognition

## Grayscale image extraction

```bash
cd data
cat raw.tar.gz.part_* > raw.tar.gz
tar -xvzf raw.tar.gz
```

## Double Random Phase Encoding

```bash
cd code/src
python crop_and_gray_images.py
python drpe_encrypt.py --src_root ../../data/grey --dst_root ../../data/drpe_encrypted --phase1_seed 42 --phase2_seed 123
```

## ConvNeXt-based Siamese Network Training and Retrieval

### DDP Multi-GPU Training

The ConvNeXt-based Siamese network supports multi-GPU distributed training with Triplet Loss.  
You can choose different ConvNeXt variants (`tiny`, `small`, `base`, `large`) using the command line.

**Install dependencies:**

```bash
pip install torch torchvision tqdm
```

**Start training (example for 4 GPUs and ConvNeXt-tiny):**

```bash
cd code/src
python train_convnext_ddp.py --gpus 0,1,2,3 --batch_size 16 --epochs 50 --convnext_variant tiny
```

- `--gpus`: Specify which GPUs to use (e.g., "0,1,2,3").
- `--batch_size`: Batch size per process (adjust according to GPU memory).
- `--epochs`: Number of training epochs.
- `--convnext_variant`: Choose from `tiny`, `small`, `base`, `large`.

The trained models will be saved to the `model/time/`, e.g., `model/run_20251016_142530/model_convnext_tiny_triplet_ddp_epoch_1.pth`.

### Retrieval with ConvNeXt Siamese

To perform retrieval (top-1/top-5 accuracy, per-class statistics):

**Run the retrieval script (example for 50 epochs and ConvNeXt-tiny):**

```bash
cd code/src
python retrieve.py --model_path ../../model/model_convnext_tiny_triplet_ddp_epoch_50.pth --convnext_variant tiny
```

- `--model_path`: Path of the model to retrieve.
- `--convnext_variant`: Choose from `tiny`, `small`, `base`, `large`.

The retrieval results (including per-class accuracy and details) will be saved in the `result/` directory as a JSON file.
