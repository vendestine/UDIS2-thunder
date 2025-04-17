#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建输出目录（如果不存在）
output_dir = os.path.join(script_dir, "extracted_fisheye_frames")
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
video_path = os.path.join(script_dir, "fisheye-4eyes.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit(1)

# 获取视频总帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频总帧数: {frame_count}")

# 询问用户要提取哪一帧（默认：第一帧）
frame_number = 0
user_input = input(f"请输入要提取的帧号 (0-{frame_count-1}, 默认=0): ")
if user_input.strip():
    frame_number = int(user_input)
    frame_number = max(0, min(frame_number, frame_count-1))  # 确保在有效范围内

# 设置帧位置
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# 读取帧
ret, frame = cap.read()
if not ret:
    print(f"错误：无法读取帧 {frame_number}")
    cap.release()
    exit(1)

# 获取尺寸
height, width = frame.shape[:2]
print(f"帧尺寸: {width}x{height}")

# 假设4个鱼眼图像按2x2网格排列
# 计算每个单独鱼眼图像的尺寸
half_width = width // 2
half_height = height // 2

# 提取4个单独的鱼眼图像
fisheye_images = [
    # 左上
    frame[0:half_height, 0:half_width],
    # 右上
    frame[0:half_height, half_width:width],
    # 左下
    frame[half_height:height, 0:half_width],
    # 右下
    frame[half_height:height, half_width:width]
]

# 保存每个鱼眼图像
for i, img in enumerate(fisheye_images):
    output_path = os.path.join(output_dir, f"fisheye_{i+1}_frame_{frame_number}.jpg")
    cv2.imwrite(output_path, img)
    print(f"已保存 {output_path}")

# 释放视频捕获
cap.release()
print(f"完成! 图像已保存到 '{output_dir}' 文件夹。") 