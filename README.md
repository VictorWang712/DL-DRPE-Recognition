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
