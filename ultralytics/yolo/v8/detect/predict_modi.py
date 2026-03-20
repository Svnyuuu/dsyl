# Ultralytics YOLO 🚀, GPL-3.0 license
# 导入系统模块
# import sys
# sys.path.append(r"D:\Work\yolo\NOW\YOLOv8-DeepSORT-Object-Tracking")
import sys  # 导入系统模块，用于操作Python运行时环境
import os  # 导入操作系统模块，用于文件和路径操作
from collections import deque, Counter  # ⭐ 添加Counter到导入中
# 获取当前文件的绝对路径
FILE = os.path.abspath(__file__)
# 获取项目根目录（向上追溯4级目录）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FILE))))
# 将项目根目录添加到系统路径中，以便导入其他模块
sys.path.append(ROOT)

import hydra  # 用于配置管理的框架
import torch  # PyTorch深度学习框架
import argparse  # 命令行参数解析
import time  # 时间相关功能
from pathlib import Path  # 面向对象的文件路径操作

from ultralytics.nn.tasks import DetectionModel  # YOLO检测模型

# 添加DetectionModel到PyTorch安全全局对象列表（用于模型加载）
torch.serialization.add_safe_globals([DetectionModel])
import torch.backends.cudnn as cudnn  # CUDA深度神经网络库
from numpy import random  # 随机数生成
# 导入Ultralytics YOLO相关模块
from ultralytics.yolo.engine.predictor import BasePredictor  # 基础预测器类
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops  # 工具函数和常量
from ultralytics.yolo.utils.checks import check_imgsz  # 检查图像大小
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box  # 绘图工具

import cv2
# 导入DeepSORT跟踪器相关模块
from deep_sort_pytorch.utils.parser import get_config  # 获取配置
from deep_sort_pytorch.deep_sort import DeepSort  # DeepSORT跟踪器
from collections import deque  # 双端队列，用于存储轨迹
import numpy as np

# 定义颜色调色板，用于为不同类别生成颜色
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# 存储每个目标的轨迹点队列
data_deque = {}
track_class_buffer = {}
# 全局DeepSORT跟踪器变量
deepsort = None


def init_tracker():
    """
    初始化DeepSORT目标跟踪器
    加载配置文件，设置跟踪参数，初始化跟踪器实例
    """
    global deepsort  # 声明使用全局变量
    cfg_deep = get_config()  # 创建配置对象
    # 从yaml文件加载DeepSORT配置
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # 创建DeepSORT跟踪器实例
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,  # 重识别模型检查点路径
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,  # 最大余弦距离
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,  # 最小置信度
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,  # NMS最大重叠
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,  # 最大IOU距离
                        max_age=cfg_deep.DEEPSORT.MAX_AGE,  # 目标最大丢失帧数
                        n_init=cfg_deep.DEEPSORT.N_INIT,  # 确认跟踪所需的帧数
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,  # 最近邻预算
                        use_cuda=True)  # 是否使用CUDA加速


##########################################################################################
def xyxy_to_xywh(*xyxy):
    """
    将xyxy格式的边界框转换为xywh（中心点坐标+宽高）格式
    xyxy: [x1, y1, x2, y2] 左上角和右下角坐标
    返回: [x_center, y_center, width, height]
    """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])  # 左边界
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])  # 上边界
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())  # 宽度
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())  # 高度
    x_c = (bbox_left + bbox_w / 2)  # 中心点x坐标
    y_c = (bbox_top + bbox_h / 2)  # 中心点y坐标
    w = bbox_w  # 宽度
    h = bbox_h  # 高度
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    """
    将xyxy格式转换为tlwh（左上角坐标+宽高）格式
    bbox_xyxy: 边界框列表，每个框为[x1,y1,x2,y2]
    返回: tlwh格式的边界框列表
    """
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]  # 转换为整数
        top = x1  # 左边界
        left = y1  # 上边界
        w = int(x2 - x1)  # 宽度
        h = int(y2 - y1)  # 高度
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    根据类别标签计算固定颜色
    label: 类别标签
    返回: BGR颜色元组
    """
    if label == 0:  # 人
        color = (85, 45, 255)
    elif label == 2:  # 汽车
        color = (222, 82, 175)
    elif label == 3:  # 摩托车
        color = (0, 204, 255)
    elif label == 5:  # 公交车
        color = (0, 149, 255)
    else:  # 其他类别使用调色板计算颜色
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    """
    绘制带有圆角边框的矩形（用于标签背景）
    img: 图像
    pt1: 左上角坐标
    pt2: 右下角坐标
    color: 颜色
    thickness: 线条粗细
    r: 圆角半径
    d: 圆角延伸距离
    """
    x1, y1 = pt1
    x2, y2 = pt2
    # 绘制四个角的圆角效果
    # 左上角
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # 右上角
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # 左下角
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # 右下角
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # 填充内部区域
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    # 绘制四个角的小圆点
    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    """
    在图像上绘制单个边界框
    x: 边界框坐标 [x1,y1,x2,y2]
    img: 图像
    color: 框的颜色
    label: 要显示的标签文本
    line_thickness: 线条粗细
    """
    # 根据图像大小动态计算线条粗细
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]  # 随机颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上角和右下角坐标
    # 绘制矩形框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        # 计算标签文本大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # 绘制带圆角的标签背景
        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)
        # 绘制标签文本
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    """
    在图像上绘制所有检测到的边界框和轨迹
    img: 图像
    bbox: 边界框列表
    names: 类别名称字典
    object_id: 目标类别ID列表
    identities: 目标跟踪ID列表
    offset: 坐标偏移量
    """
    height, width, _ = img.shape

    # 移除丢失的目标的轨迹点
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    # 遍历每个检测到的目标
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # 应用偏移量
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # 计算底部边缘中心点（用于轨迹）
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # 获取目标的跟踪ID
        id = int(identities[i]) if identities is not None else 0

        # 为新目标创建轨迹缓冲区
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)  # 最多保留64个轨迹点

        # 根据类别ID获取颜色
        color = compute_color_for_labels(object_id[i])
        # 获取类别名称
        obj_name = names[object_id[i]]
        # 创建标签文本（跟踪ID:类别名称）
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # 将当前中心点添加到轨迹缓冲区
        data_deque[id].appendleft(center)

        # 绘制边界框和标签
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # 绘制轨迹线
        for i in range(1, len(data_deque[id])):
            # 跳过无效点
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # 动态计算线条粗细（越近的点线条越粗）
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # 绘制轨迹线段
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img


class DetectionPredictor(BasePredictor):
    """
    继承自YOLO的BasePredictor，实现目标检测和跟踪的预测器
    """

    def get_annotator(self, img):
        """
        获取标注器实例
        img: 输入图像
        返回: Annotator对象
        """
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        """
        图像预处理：转换为tensor、归一化
        img: 输入图像
        返回: 预处理后的tensor
        """
        img = torch.from_numpy(img).to(self.model.device)  # 转换为tensor并移到指定设备
        img = img.half() if self.model.fp16 else img.float()  # 根据模型精度转换数据类型
        img /= 255  # 归一化到[0,1]
        return img

    def postprocess(self, preds, img, orig_img):
        """
        后处理：NMS和边界框缩放
        preds: 模型原始预测结果
        img: 预处理后的图像
        orig_img: 原始图像
        返回: 后处理后的预测结果
        """
        # 执行非极大值抑制
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,  # 置信度阈值
                                        self.args.iou,  # IOU阈值
                                        agnostic=self.args.agnostic_nms,  # 是否类别无关的NMS
                                        max_det=self.args.max_det)  # 最大检测数

        # 将边界框坐标缩放到原始图像尺寸
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        """
        写入检测和跟踪结果到图像
        idx: 批次索引
        preds: 预测结果
        batch: 批次数据 (p, im, im0)
        返回: 日志字符串
        """
        p, im, im0 = batch  # 解包批次数据：路径、处理后的图像、原始图像
        all_outputs = []
        log_string = ""

        # 处理批次维度
        if len(im.shape) == 3:
            im = im[None]  # 扩展批次维度

        self.seen += 1  # 已处理的图像计数
        im0 = im0.copy()  # 复制原始图像，避免修改原图

        # 获取帧信息
        if self.webcam:  # 网络摄像头模式
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:  # 视频文件模式
            frame = getattr(self.dataset, 'frame', 0)

        # 设置保存路径
        self.data_path = p
        save_path = str(self.save_dir / p.name)  # 图像保存路径
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')

        log_string += '%gx%g ' % im.shape[2:]  # 记录图像尺寸
        self.annotator = self.get_annotator(im0)  # 获取标注器

        det = preds[idx]  # 获取当前批次的检测结果
        all_outputs.append(det)

        if len(det) == 0:  # 没有检测到目标
            return log_string

        # 统计每个类别的检测数量
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # 当前类别的检测数
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # 准备DeepSORT输入数据
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益
        xywh_bboxs = []  # 存储xywh格式的边界框
        confs = []  # 存储置信度
        oids = []  # 存储目标类别ID
        outputs = []  # 存储跟踪结果

        h_img, w_img = im0.shape[:2]  # <--- 获取当前图像的高和宽

        # 遍历每个检测结果
        for *xyxy, conf, cls in reversed(det):
            # <--- 新增过滤逻辑 开始 --->
            x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
            conf_val = conf.item()

            # 1. 过滤路牌误检：过滤画面上半部分(30%)且置信度较低的车辆/卡车
            if y1 < int(h_img * 0.3) and conf_val < 0.55:
                continue

            # 2. 边缘过滤：框贴近画面边缘(15像素内)时，通常是不完整的车，容易造成巨型框，直接丢弃
            margin = 15
            if x1 < margin or y1 < margin or x2 > (w_img - margin) or y2 > (h_img - margin):
                continue

            # 3. 异常长宽比过滤：宽是高4倍以上，或高是宽4倍以上的畸形框直接丢弃
            box_width = x2 - x1
            box_height = max(y2 - y1, 1)  # 防止除以0
            if (box_width / box_height > 4.0) or (box_height / box_width > 4.0):
                continue
            # <--- 新增过滤逻辑 结束 --->

            # 转换为xywh格式（中心点坐标+宽高）
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))

        # 转换为tensor
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # 使用DeepSORT进行目标跟踪
        outputs = deepsort.update(xywhs, confss, oids, im0)

        # 如果有跟踪结果，绘制边界框和轨迹
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]  # 跟踪得到的边界框
            identities = outputs[:, -2]  # 跟踪ID
            object_id = outputs[:, -1]  # 类别ID

            # <--- 新增类别平滑逻辑 开始 --->
            # 遍历当前画面中的所有追踪目标
            for i, track_id in enumerate(identities):
                tid = int(track_id)
                cid = int(object_id[i])

                # 如果这个ID是第一次出现，给它分配一个容量为5的队列
                if tid not in track_class_buffer:
                    track_class_buffer[tid] = deque(maxlen=5)

                # 将YOLO这一帧判断的类别加入队列
                track_class_buffer[tid].append(cid)

                # 核心：多数投票！取最近5帧里出现次数最多的类别作为最终类别
                stable_cid = Counter(track_class_buffer[tid]).most_common(1)[0][0]

                # 覆盖原本闪烁不定的类别ID
                object_id[i] = stable_cid
            # <--- 新增类别平滑逻辑 结束 --->

            # 在图像上绘制边界框和轨迹 (此时传入的 object_id 已经是平滑过后的了)
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    """
    主预测函数，使用Hydra配置管理
    cfg: 配置对象
    """
    init_tracker()  # 初始化DeepSORT跟踪器
    # cfg.model = cfg.model or "yolov8n.pt"  # 设置默认模型
    cfg.model = cfg.model or "yolov8m.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # 检查并调整图像大小
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"  # 设置默认输入源
    predictor = DetectionPredictor(cfg)  # 创建预测器实例
    predictor()  # 执行预测


if __name__ == "__main__":
    predict()  # 程序入口