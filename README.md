# DL-DRPE-Recognition

## Encrypt

```bash
cd data
tar -xJf ./raw.tar.xz

cd ../src
python crop_and_gray_images.py
python drpe_encrypt.py --src_root ../../data/128px_grey --dst_root ../../data/drpe_encrypted --phase1_seed 42 --phase2_seed 123
```
