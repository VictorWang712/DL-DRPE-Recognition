# DL-DRPE-Recognition

## Grayscale image extraction

```bash
cd data
unzip raw.zip
```

## Double Random Phase Encoding

```bash
cd code/src
python crop_and_gray_images.py
python drpe_encrypt.py --src_root ../../data/grey --dst_root ../../data/drpe_encrypted --phase1_seed 42 --phase2_seed 123
```
