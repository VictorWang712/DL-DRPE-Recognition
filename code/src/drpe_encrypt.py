import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def drpe_encrypt(img_array, phase1, phase2):
    # img_array: 输入灰度图像，归一化到[0,1]
    # phase1, phase2: 两个随机相位矩阵
    # 1. 空间域相位调制
    img_mod = img_array * np.exp(1j * phase1)
    # 2. 傅里叶变换
    img_fft = np.fft.fft2(img_mod)
    # 3. 频域相位调制
    img_fft_mod = img_fft * np.exp(1j * phase2)
    # 4. 逆傅里叶变换
    encrypted = np.fft.ifft2(img_fft_mod)
    # 返回复数密文
    return encrypted

def save_encrypted_image(enc_img, save_path):
    # 保存密文为npy文件
    np.save(save_path, enc_img)

def save_visualization(enc_img, save_path_prefix):
    # 幅值（模）归一化到[0,255]保存为PNG
    mag = np.abs(enc_img)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    Image.fromarray(mag).save(save_path_prefix + '_mag.png')

    # 相位归一化到[0,255]保存为PNG
    phase = np.angle(enc_img)
    phase = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    Image.fromarray(phase).save(save_path_prefix + '_phase.png')

def load_image(img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0
    return img

def generate_phase(shape, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=shape)

def main(
    src_root = '../../data/128px_grey',
    dst_root = '../../data/drpe_encrypted',
    phase1_seed = 42,
    phase2_seed = 123
):
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), src_root))
    dst_root = os.path.abspath(os.path.join(os.path.dirname(__file__), dst_root))
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # 预生成相位掩模（假设所有图片尺寸一致）
    sample_class = os.listdir(src_root)[0]
    sample_img_path = os.path.join(src_root, sample_class, os.listdir(os.path.join(src_root, sample_class))[0])
    img_shape = load_image(sample_img_path).shape
    phase1 = generate_phase(img_shape, phase1_seed)
    phase2 = generate_phase(img_shape, phase2_seed)

    # 保存相位掩模，方便解密
    np.save(os.path.join(dst_root, 'phase1.npy'), phase1)
    np.save(os.path.join(dst_root, 'phase2.npy'), phase2)

    for cls in tqdm(os.listdir(src_root), desc='Classes'):
        src_cls_dir = os.path.join(src_root, cls)
        dst_cls_dir = os.path.join(dst_root, cls)
        if not os.path.isdir(src_cls_dir):
            continue
        if not os.path.exists(dst_cls_dir):
            os.makedirs(dst_cls_dir)
        for fname in tqdm(os.listdir(src_cls_dir), desc=cls, leave=False):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                continue
            src_img_path = os.path.join(src_cls_dir, fname)
            dst_img_path = os.path.join(dst_cls_dir, fname + '.npy')
            img = load_image(src_img_path)
            enc_img = drpe_encrypt(img, phase1, phase2)
            save_encrypted_image(enc_img, dst_img_path)

            # 保存可视化图片
            save_path_prefix = os.path.join(dst_cls_dir, fname)
            save_visualization(enc_img, save_path_prefix)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DRPE Encrypt Images')
    parser.add_argument('--src_root', type=str, default='../../data/128px_grey', help='Source image root')
    parser.add_argument('--dst_root', type=str, default='../../data/drpe_encrypted', help='Encrypted image root')
    parser.add_argument('--phase1_seed', type=int, default=42, help='Seed for spatial phase')
    parser.add_argument('--phase2_seed', type=int, default=123, help='Seed for frequency phase')
    args = parser.parse_args()
    main(args.src_root, args.dst_root, args.phase1_seed, args.phase2_seed)
