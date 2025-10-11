import os
from PIL import Image

# 源数据文件夹和目标文件夹
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
DST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processed'))

def process_image(src_path, dst_path):
    try:
        img = Image.open(src_path)
        # 等比例缩放，短边=128
        w, h = img.size
        if w < h:
            new_w = 128
            new_h = int(h * (128 / w))
        else:
            new_h = 128
            new_w = int(w * (128 / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # 从左上角裁剪128x128
        img = img.crop((0, 0, 128, 128))
        # 转为灰度
        img = img.convert('L')
        # 保存
        img.save(dst_path)
    except Exception as e:
        print(f"处理图片 {src_path} 时出错: {e}")

def process_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)
        if os.path.isfile(src_path) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            process_image(src_path, dst_path)

def main():
    for subdir in os.listdir(SRC_ROOT):
        src_subdir = os.path.join(SRC_ROOT, subdir)
        dst_subdir = os.path.join(DST_ROOT, subdir)
        if os.path.isdir(src_subdir):
            print(f"处理文件夹: {subdir}")
            process_folder(src_subdir, dst_subdir)

if __name__ == '__main__':
    main()
