# /data/hdd1/kb/MyProjects/ml-depth-pro/get_depth_images.py

import os
import re
import numpy as np
import cv2
from PIL import Image
import depth_pro
import torch

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()

# 输入和输出文件夹路径
input_folder = "/data/net/dl_data/ProjectDatasets_bkx/pick_and_place_dataset/armbench-segmentation-0.1/mix-object-tote/images"
output_folder = "/data/net/dl_data/ProjectDatasets_bkx/pick_and_place_dataset/armbench-segmentation-0.1/depth_mix-object-tote/images"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def save_depth_as_png(depth_array, output_path):
    """将深度数据转换为jet colormap并保存为PNG格式"""
    depth_min, depth_max = np.min(depth_array), np.max(depth_array)
    if depth_max - depth_min > 1e-6:  # 避免除零错误
        depth_norm = (depth_array - depth_min) / (depth_max - depth_min)  # 归一化到 [0,1]
    else:
        depth_norm = np.zeros_like(depth_array)  # 处理全0情况

    depth_colored = (depth_norm * 255).astype(np.uint8)  # 归一化到 [0,255]
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)  # 使用 Jet colormap

    cv2.imwrite(output_path, depth_colored)  # 保存为 PNG 格式


def output_files_exist(output_folder, file_base):
    """检查是否已存在匹配的深度文件"""
    for file in os.listdir(output_folder):
        if file.startswith(f"depth_{file_base}"):
            return True
    return False


# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
        image_path = os.path.join(input_folder, filename)
        
        file_base = os.path.splitext(filename)[0]  # 获取去掉扩展名的文件名
        # 预检查是否已经处理过
        if output_files_exist(output_folder, file_base):
            print(f"Skipping {filename}, output already exists.")
            continue

        try:
            # 加载和预处理图像
            image, _, f_px = depth_pro.load_rgb(image_path)
            image = transform(image)

            # 运行推理
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"].cpu().numpy()  # 转换为 NumPy 数组
            focallength_px = prediction["focallength_px"].item()  # 转换为数值
            
            # **去掉文件扩展名**，避免重复 `.jpg.png`             
            file_base = os.path.splitext(filename)[0]  # 获取去掉扩展名的文件名
            
            # 构造输出文件名
            depth_npz_filename = f"depth_{file_base}_{focallength_px}.npz"
            depth_png_filename = f"depth_{file_base}_{focallength_px}.png"
            npz_output_path = os.path.join(output_folder, depth_npz_filename)
            png_output_path = os.path.join(output_folder, depth_png_filename)

            # 保存深度信息为 npy
            np.savez_compressed(npz_output_path, depth=depth)

            # 保存可视化深度图
            save_depth_as_png(depth, png_output_path)

            print(f"Saved: {npz_output_path}")
            print(f"Saved: {png_output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Processing complete.")

