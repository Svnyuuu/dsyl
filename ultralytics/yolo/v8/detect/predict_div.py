# Ultralytics YOLO 🚀

import sys
import os

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FILE))))
sys.path.append(ROOT)

import hydra
import torch
import time
from pathlib import Path
import cv2
import numpy as np

from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from collections import deque

# ================= 全局变量 =================
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

# 计数
counted_ids = set()
enter_count = {"car": 0, "truck": 0, "bus": 0}
leave_count = {"car": 0, "truck": 0, "bus": 0}

# 计数区域参数
line_y = 0
count_zone_thickness = 30  # 计数区域的厚度（像素）

# ================= 新增：机动车道区域定义 =================
# 定义机动车道区域（多边形）
# 格式：[(x1,y1), (x2,y2), (x3,y3), ...] 顺时针或逆时针
# 你需要根据实际视频调整这些坐标
road_polygon = np.array([
    [100, 300],  # 左上角
    [900, 300],  # 右上角
    [900, 700],  # 右下角
    [100, 700]  # 左下角
], np.int32)

# 过滤置信度阈值
min_confidence = 0.5  # 低于此值的检测将被忽略

# 有效类别（只统计这些类别）
valid_classes = ["car", "truck", "bus"]

deepsort = None


# ================= 初始化 DeepSORT =================
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )


# ================= 新增：区域检查函数 =================
def is_point_in_road(point, polygon):
    """检查点是否在多边形区域内"""
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def is_bbox_in_road(bbox, polygon, threshold=0.3):
    """
    检查边界框是否在道路区域内
    bbox: [x1, y1, x2, y2]
    threshold: 框需要有多少面积在道路内才被认为是有效的 (0-1)
    """
    x1, y1, x2, y2 = bbox

    # 计算框的中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = (int(center_x), int(center_y))

    # 检查中心点是否在区域内
    if is_point_in_road(center_point, polygon):
        return True, "center"

    # 可选：检查框的底部中心点（更适合车辆）
    bottom_center = (int(center_x), int(y2))
    if is_point_in_road(bottom_center, polygon):
        return True, "bottom"

    return False, None


def draw_road_polygon(img, polygon):
    """绘制道路区域"""
    overlay = img.copy()

    # 绘制半透明道路区域
    cv2.fillPoly(overlay, [polygon], (0, 255, 0))  # 绿色区域
    alpha = 0.1  # 低透明度
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 绘制道路边界
    cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

    # 添加文字
    cv2.putText(img, "ROAD AREA", (polygon[0][0], polygon[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return img


# ================= 工具函数 =================
def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h


def compute_color_for_labels(label):
    if label == 2:  # car
        return (222, 82, 175)
    elif label == 5:  # bus
        return (0, 149, 255)
    elif label == 7:  # truck
        return (255, 204, 0)
    else:
        return (255, 255, 255)


def UI_box(x, img, color=None, label=None, line_thickness=2, confidence=None):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # 根据置信度调整边框颜色
    if confidence:
        if confidence < 0.5:
            color = (128, 128, 128)  # 灰色表示低置信度
        elif confidence < 0.7:
            color = (0, 165, 255)  # 橙色表示中等置信度

    cv2.rectangle(img, c1, c2, color, thickness=line_thickness)

    if label:
        # 添加置信度信息
        if confidence:
            label = f"{label} ({confidence:.2f})"
        cv2.putText(img, label, (c1[0], c1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_count_zone(img, line_y, zone_thickness):
    """绘制计数区域（可视化检测带）"""
    h, w, _ = img.shape

    # 计算区域边界
    zone_top = max(0, line_y - zone_thickness // 2)
    zone_bottom = min(h, line_y + zone_thickness // 2)

    # 创建半透明覆盖层
    overlay = img.copy()

    # 绘制半透明计数区域（浅蓝色）
    cv2.rectangle(overlay, (0, zone_top), (w, zone_bottom), (255, 255, 0), -1)  # 黄色区域

    # 混合透明度
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 绘制边界线
    cv2.line(img, (0, line_y), (w, line_y), (0, 255, 0), 2)  # 中心线（绿色）
    cv2.line(img, (0, zone_top), (w, zone_top), (0, 255, 255), 1)  # 上边界（黄色）
    cv2.line(img, (0, zone_bottom), (w, zone_bottom), (0, 255, 255), 1)  # 下边界（黄色）

    # 添加文字说明
    cv2.putText(img, "COUNTING ZONE", (w // 2 - 100, zone_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img


# ================= 核心：画框 + 计数 =================
def draw_boxes(img, bbox, names, object_id, identities, detection_info=None):
    global line_y

    h, w, _ = img.shape
    line_y = int(h * 0.6)

    # 绘制道路区域
    img = draw_road_polygon(img, road_polygon)

    # 绘制计数区域
    img = draw_count_zone(img, line_y, count_zone_thickness)

    # 清理丢失ID
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    # 统计在道路内外的目标
    road_objects = []
    outside_objects = []

    # 创建检测信息的字典，方便查找
    det_info_dict = {}
    if detection_info:
        for info in detection_info:
            track_id = info.get('track_id')
            if track_id is not None:
                det_info_dict[track_id] = info

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = map(int, box)

        track_id = int(identities[i])
        cls_id = int(object_id[i])
        obj_name = names[cls_id]

        # 获取该track_id对应的检测信息
        det_info = det_info_dict.get(track_id, {})
        confidence = det_info.get('confidence', 1.0)
        original_box = det_info.get('original_box', box)

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        bottom_center = (int((x1 + x2) / 2), int(y2))

        # 过滤低置信度检测
        if confidence < min_confidence:
            continue

        # 检查是否在道路区域内
        in_road, check_type = is_bbox_in_road([x1, y1, x2, y2], road_polygon)

        # 过滤不在道路区域内的非车辆目标
        if obj_name not in valid_classes:
            continue

        if track_id not in data_deque:
            data_deque[track_id] = deque(maxlen=30)

        data_deque[track_id].appendleft(center)

        # ===== 过线判断（只在道路区域内进行）=====
        if in_road and len(data_deque[track_id]) >= 2:
            prev_y = data_deque[track_id][1][1]
            curr_y = data_deque[track_id][0][1]

            if track_id not in counted_ids:
                # 进入
                if prev_y < line_y and curr_y >= line_y:
                    enter_count[obj_name] += 1
                    counted_ids.add(track_id)
                    road_objects.append((track_id, obj_name, "entering"))

                # 离开
                elif prev_y > line_y and curr_y <= line_y:
                    leave_count[obj_name] += 1
                    counted_ids.add(track_id)
                    road_objects.append((track_id, obj_name, "leaving"))

        # ===== 根据不同情况绘制不同样式的框 =====
        color = compute_color_for_labels(cls_id)

        # 标记是否在计数区域内
        zone_top = max(0, line_y - count_zone_thickness // 2)
        zone_bottom = min(h, line_y + count_zone_thickness // 2)

        if not in_road:
            # 不在道路区域内 - 显示为半透明灰色
            color = (128, 128, 128)
            label = f"{track_id}:{obj_name} [OUTSIDE]"
            UI_box(box, img, color, label, line_thickness=1, confidence=confidence)
            # 添加"非道路区域"标记
            cv2.putText(img, "NOT IN ROAD", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            outside_objects.append((track_id, obj_name))
        elif zone_top <= center[1] <= zone_bottom:
            # 在道路内且处于计数区域
            label = f"{track_id}:{obj_name}"
            if check_type == "bottom":
                label += " [BOTTOM]"
            UI_box(box, img, color, label, line_thickness=3, confidence=confidence)
            cv2.circle(img, bottom_center, 5, (0, 255, 255), -1)  # 底部中心点
            road_objects.append((track_id, obj_name, "in_zone"))
        else:
            # 在道路内但不在计数区域
            label = f"{track_id}:{obj_name}"
            UI_box(box, img, color, label, line_thickness=2, confidence=confidence)

        # ===== 轨迹（只绘制道路内的轨迹）=====
        if in_road:
            for j in range(1, len(data_deque[track_id])):
                cv2.line(img,
                         data_deque[track_id][j - 1],
                         data_deque[track_id][j],
                         color, 2)

    # ===== 显示统计信息 =====
    # 左边显示进入统计
    y = 40
    cv2.rectangle(img, (10, 25), (300, 25 + len(enter_count) * 30), (0, 0, 0), -1)
    for k, v in enter_count.items():
        cv2.putText(img, f"ENTERING {k}:{v}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30

    # 右边显示离开统计
    y = 40
    cv2.rectangle(img, (w - 310, 25), (w - 10, 25 + len(leave_count) * 30), (0, 0, 0), -1)
    for k, v in leave_count.items():
        cv2.putText(img, f"LEAVING {k}:{v}", (w - 300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 30

    # 显示当前统计
    cv2.putText(img, f"Objects in road: {len(road_objects)}", (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Objects outside: {len(outside_objects)}", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, f"Min confidence: {min_confidence}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# ================= Predictor =================
class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.float() / 255.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        im0 = im0.copy()

        det = preds[idx]
        if len(det) == 0:
            return ""

        xywh_bboxs, confs, oids = [], [], []
        detection_info = []  # 保存每个检测的详细信息

        for *xyxy, conf, cls in det:
            x_c, y_c, w, h = xyxy_to_xywh(*xyxy)
            xywh_bboxs.append([x_c, y_c, w, h])
            confs.append([conf.item()])
            oids.append(int(cls))

            # 保存原始检测信息，后面会通过track_id关联
            detection_info.append({
                'confidence': conf.item(),
                'class': int(cls),
                'bbox': [x.item() for x in xyxy],
                'track_id': None  # 将在后面更新
            })

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            # 更新detection_info中的track_id
            for i, track_id in enumerate(identities):
                if i < len(detection_info):
                    detection_info[i]['track_id'] = int(track_id)

            # 传递检测信息
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, detection_info)

        self.plotted_img = im0
        return ""


# ================= 主函数 =================
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict_div(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8m.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)

    # 设置置信度阈值
    cfg.conf = 0.3  # YOLO的初始置信度阈值

    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict_div()