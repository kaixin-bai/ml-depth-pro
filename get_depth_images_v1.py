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

# 获取输入文件夹中的所有基础文件名
input_files = {os.path.splitext(f)[0] for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

# 获取输出文件夹中已有的深度文件对应的基础文件名
existing_files = set()
for f in os.listdir(output_folder):
    match = re.match(r"depth_([^_]*)_.*\.npz", f)
    if match:
        existing_files.add(match.group(1))

if existing_files:
    print(f"First existing file base: {next(iter(existing_files))}")

# 找出需要处理的文件
files_to_process = input_files - existing_files
print(f"files_to_process: {len(files_to_process)}")
print(f"input_files: {len(input_files)}")
print(f"existing_filess: {len(existing_files)}")

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

# 遍历需要处理的文件
for file_base in files_to_process:
    image_path = os.path.join(input_folder, file_base + ".jpg")  # 假设输入文件是 PNG，若可能有 JPG 需修改
    
    try:
        # 加载和预处理图像
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)

        # 运行推理
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"].cpu().numpy()  # 转换为 NumPy 数组
        focallength_px = prediction["focallength_px"].item()  # 转换为数值
        
        # 生成文件名
        depth_npz_filename = f"depth_{file_base}_{focallength_px}.npz"
        depth_png_filename = f"depth_{file_base}_{focallength_px}.png"
        npz_output_path = os.path.join(output_folder, depth_npz_filename)
        png_output_path = os.path.join(output_folder, depth_png_filename)

        # 保存深度信息为 npz
        np.savez_compressed(npz_output_path, depth=depth)
        
        # 保存可视化深度图
        save_depth_as_png(depth, png_output_path)

        print(f"Saved: {npz_output_path}")
        print(f"Saved: {png_output_path}")

    except Exception as e:
        print(f"Error processing {file_base}: {e}")

print("Processing complete.")

