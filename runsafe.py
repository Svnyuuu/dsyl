# runsafe.py
import torch
from pathlib import Path
from ultralytics import YOLO

# -------------------------------
# YOLOv8 + DeepSORT 权重反序列化允许列表
# -------------------------------
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU

# 安全允许全局类
torch.serialization.add_safe_globals([
    DetectionModel, Conv, Sequential, Conv2d, BatchNorm2d, SiLU
])

# -------------------------------
# 配置权重文件
# -------------------------------
YOLO_WEIGHTS = "yolov8l.pt"          # YOLOv8 权重
VIDEO_SOURCE = "test3.mp4"           # 视频路径
SHOW_VIDEO = True                     # 是否显示视频

# -------------------------------
# 初始化 YOLOv8 模型
# -------------------------------
print("正在加载 YOLOv8 模型，请稍等...")
model = YOLO(YOLO_WEIGHTS)
print("YOLOv8 模型加载成功！")

# -------------------------------
# 运行视频检测 + DeepSORT
# -------------------------------
# 这里使用 predict API，可以直接用 DeepSORT 集成后的版本
results = model.predict(
    source=VIDEO_SOURCE,
    show=SHOW_VIDEO,    # 显示窗口
    stream=False,       # 如果 True 会逐帧返回 generator
)

print("检测完成！")