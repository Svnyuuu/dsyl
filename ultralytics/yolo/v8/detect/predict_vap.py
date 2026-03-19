# Ultralytics YOLO 🚀
# 同时识别车和人 vehicle and pedestrians
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
import argparse

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
# ⭐ 修改1：扩展计数字典，包含人和各类车辆
enter_count = {
    "person": 0,    # 行人
    "car": 0,       # 汽车
    "truck": 0,     # 卡车
    "bus": 0,       # 公交车
    "motorcycle": 0,# 摩托车
    "bicycle": 0    # 自行车
}
leave_count = {
    "person": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "bicycle": 0
}

# 计数区域参数
line_y = 0
count_zone_thickness = 30

# ================= 新增：交互式多边形定义 =================
road_polygon = None
drawing = False
points = []
deepsort = None

# ⭐ 修改2：有效类别 - 同时包括人和车
valid_classes = [
    "person",        # 人 (类别ID: 0)
    "car",           # 汽车 (类别ID: 2)
    "truck",         # 卡车 (类别ID: 7)
    "bus",           # 公交车 (类别ID: 5)
    "motorcycle",    # 摩托车 (类别ID: 3)
    "bicycle"        # 自行车 (类别ID: 1)
]

# YOLO类别ID到名称的映射
class_id_to_name = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

min_confidence = 0.5


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


# ================= 鼠标回调函数 =================
def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于绘制多边形"""
    global points, road_polygon, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键点击添加点
        points.append((x, y))
        print(f"添加点: ({x}, {y})")

        # 在图像上画点
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
        if len(points) > 1:
            # 画线连接上一个点
            cv2.line(img_copy, points[-2], points[-1], (0, 255, 0), 2)

        # 如果已经点了4个点，自动闭合多边形
        if len(points) == 4:
            cv2.line(img_copy, points[-1], points[0], (0, 255, 0), 2)
            road_polygon = np.array(points, np.int32)
            print(f"多边形完成！坐标: {points}")
            print("按 's' 保存并继续，按 'r' 重新绘制")

        cv2.imshow("Draw Road Area - Click 4 corners", img_copy)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键点击清除最后一个点
        if points:
            points.pop()
            print("移除最后一个点")
            # 重新绘制
            refresh_drawing()


def refresh_drawing():
    """刷新绘图窗口"""
    global img_copy, points
    img_copy = original_img.copy()

    # 重新绘制所有点
    for i, point in enumerate(points):
        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(img_copy, points[i - 1], points[i], (0, 255, 0), 2)

    # 如果已经有4个点，画闭合线
    if len(points) == 4:
        cv2.line(img_copy, points[-1], points[0], (0, 255, 0), 2)

    cv2.imshow("Draw Road Area - Click 4 corners", img_copy)


def get_road_polygon_from_user(first_frame):
    """让用户绘制道路区域多边形"""
    global original_img, img_copy, points, road_polygon

    original_img = first_frame.copy()
    img_copy = first_frame.copy()
    points = []
    road_polygon = None

    cv2.namedWindow("Draw Road Area - Click 4 corners")
    cv2.setMouseCallback("Draw Road Area - Click 4 corners", mouse_callback)

    print("\n=== 请绘制道路区域 ===")
    print("1. 依次点击4个角点（顺时针或逆时针）")
    print("2. 右键点击可以撤销上一个点")
    print("3. 点击完4个点后，按 's' 保存并继续")
    print("4. 按 'r' 重新绘制")
    print("5. 按 'q' 退出程序\n")

    cv2.imshow("Draw Road Area - Click 4 corners", img_copy)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and road_polygon is not None:
            # 保存多边形
            cv2.destroyWindow("Draw Road Area - Click 4 corners")
            print(f"道路区域已保存: {points}")
            return road_polygon

        elif key == ord('r'):
            # 重新绘制
            points = []
            road_polygon = None
            img_copy = original_img.copy()
            cv2.imshow("Draw Road Area - Click 4 corners", img_copy)
            print("重新开始绘制...")

        elif key == ord('q'):
            # 退出
            cv2.destroyAllWindows()
            sys.exit("用户退出程序")


# ================= 区域检查函数 =================
def is_point_in_road(point, polygon):
    """检查点是否在多边形区域内"""
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def is_bbox_in_road(bbox, polygon):
    """检查边界框是否在道路区域内"""
    if polygon is None:
        return True, "none"

    x1, y1, x2, y2 = bbox

    # 计算框的中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_point = (int(center_x), int(center_y))

    # 检查中心点是否在区域内
    if is_point_in_road(center_point, polygon):
        return True, "center"

    # 检查框的底部中心点（对于行人，这个点更准确）
    bottom_center = (int(center_x), int(y2))
    if is_point_in_road(bottom_center, polygon):
        return True, "bottom"

    return False, None


def draw_road_polygon(img, polygon):
    """绘制道路区域"""
    if polygon is None:
        return img

    overlay = img.copy()

    # 绘制半透明道路区域
    cv2.fillPoly(overlay, [polygon], (0, 255, 0))
    alpha = 0.1
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 绘制道路边界
    cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

    # 在角点标注坐标
    for i, point in enumerate(polygon):
        cv2.circle(img, tuple(point), 5, (0, 255, 255), -1)
        cv2.putText(img, f"{i + 1}", (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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


# ⭐ 修改3：扩展颜色函数，为人和所有车辆类别分配不同颜色
def compute_color_for_labels(label):
    """
    根据类别标签计算固定颜色
    """
    if label == 0:  # person
        return (255, 255, 0)      # 青色
    elif label == 1:  # bicycle
        return (255, 0, 255)      # 洋红色
    elif label == 2:  # car
        return (222, 82, 175)     # 紫色
    elif label == 3:  # motorcycle
        return (0, 204, 255)      # 橙色
    elif label == 5:  # bus
        return (0, 149, 255)      # 蓝色
    elif label == 7:  # truck
        return (255, 204, 0)      # 黄色
    else:
        return (255, 255, 255)    # 白色（默认）


def UI_box(x, img, color=None, label=None, line_thickness=2, confidence=None):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    if confidence:
        if confidence < 0.5:
            color = (128, 128, 128)
        elif confidence < 0.7:
            color = (0, 165, 255)

    cv2.rectangle(img, c1, c2, color, thickness=line_thickness)

    if label:
        if confidence:
            label = f"{label} ({confidence:.2f})"
        cv2.putText(img, label, (c1[0], c1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_count_zone(img, line_y, zone_thickness):
    """绘制计数区域"""
    h, w, _ = img.shape

    zone_top = max(0, line_y - zone_thickness // 2)
    zone_bottom = min(h, line_y + zone_thickness // 2)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, zone_top), (w, zone_bottom), (255, 255, 0), -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.line(img, (0, line_y), (w, line_y), (0, 255, 0), 2)
    cv2.line(img, (0, zone_top), (w, zone_top), (0, 255, 255), 1)
    cv2.line(img, (0, zone_bottom), (w, zone_bottom), (0, 255, 255), 1)

    cv2.putText(img, "COUNTING ZONE", (w // 2 - 100, zone_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img


# ================= 核心：画框 + 计数（只显示区域内的人和车） =================
def draw_boxes(img, bbox, names, object_id, identities, detection_info=None):
    global line_y

    h, w, _ = img.shape
    line_y = int(h * 0.6)

    # 绘制道路区域（半透明，方便看到区域范围）
    img = draw_road_polygon(img, road_polygon)

    # 绘制计数区域
    img = draw_count_zone(img, line_y, count_zone_thickness)

    # 清理丢失ID
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    road_objects = []

    # 创建检测信息的字典
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

        det_info = det_info_dict.get(track_id, {})
        confidence = det_info.get('confidence', 1.0)

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        bottom_center = (int((x1 + x2) / 2), int(y2))

        # 过滤低置信度
        if confidence < min_confidence:
            continue

        # ⭐ 修改4：检查是否在道路区域内（人和车都检查）
        in_road, check_type = is_bbox_in_road([x1, y1, x2, y2], road_polygon)

        # ⭐ 修改5：只处理有效类别（人和车）且在道路内的目标
        if obj_name not in valid_classes or not in_road:
            continue  # 不在道路内或不感兴趣的类别直接跳过

        # 只有道路内的目标才会执行到这里
        if track_id not in data_deque:
            data_deque[track_id] = deque(maxlen=30)

        data_deque[track_id].appendleft(center)

        # 过线判断
        if len(data_deque[track_id]) >= 2:
            prev_y = data_deque[track_id][1][1]
            curr_y = data_deque[track_id][0][1]

            if track_id not in counted_ids:
                if prev_y < line_y and curr_y >= line_y:
                    enter_count[obj_name] += 1
                    counted_ids.add(track_id)
                    road_objects.append((track_id, obj_name, "entering"))
                elif prev_y > line_y and curr_y <= line_y:
                    leave_count[obj_name] += 1
                    counted_ids.add(track_id)
                    road_objects.append((track_id, obj_name, "leaving"))

        # 绘制框
        color = compute_color_for_labels(cls_id)

        zone_top = max(0, line_y - count_zone_thickness // 2)
        zone_bottom = min(h, line_y + count_zone_thickness // 2)

        # ⭐ 修改6：根据类别显示不同的标签
        if obj_name == "person":
            # 行人的标签特殊处理，用更醒目的方式
            label = f"P{track_id}:{obj_name}"
        else:
            # 车辆的标签
            label = f"V{track_id}:{obj_name}"

        if zone_top <= center[1] <= zone_bottom:
            # 在计数区域内
            if check_type == "bottom":
                label += " [BOTTOM]"
            UI_box(box, img, color, label, line_thickness=3, confidence=confidence)
            cv2.circle(img, bottom_center, 5, (0, 255, 255), -1)
            road_objects.append((track_id, obj_name, "in_zone"))
        else:
            # 在道路内但不在计数区域
            UI_box(box, img, color, label, line_thickness=2, confidence=confidence)

        # 绘制轨迹
        for j in range(1, len(data_deque[track_id])):
            cv2.line(img,
                     data_deque[track_id][j - 1],
                     data_deque[track_id][j],
                     color, 2)

    # ⭐ 修改7：显示统计信息（包括人和各类车辆）
    # 左侧统计（进入的）
    y = 40
    cv2.rectangle(img, (10, 25), (300, 25 + len(enter_count) * 30), (0, 0, 0), -1)
    cv2.putText(img, "ENTERING", (20, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for k, v in enter_count.items():
        # 只显示有计数的类别
        if v > 0 or k in ["person", "car"]:  # 至少显示人和车
            color = (0, 255, 0) if k == "person" else (0, 255, 0)
            cv2.putText(img, f"{k}:{v}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

    # 右侧统计（离开的）
    y = 40
    cv2.rectangle(img, (w - 310, 25), (w - 10, 25 + len(leave_count) * 30), (0, 0, 0), -1)
    cv2.putText(img, "LEAVING", (w - 300, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for k, v in leave_count.items():
        if v > 0 or k in ["person", "car"]:
            color = (0, 0, 255) if k == "person" else (0, 0, 255)
            cv2.putText(img, f"{k}:{v}", (w - 300, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

    # 显示道路内目标总数
    person_count = sum(1 for obj in road_objects if obj[1] == "person")
    vehicle_count = len(road_objects) - person_count
    cv2.putText(img, f"Pedestrians: {person_count} | Vehicles: {vehicle_count}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
        detection_info = []

        for *xyxy, conf, cls in det:
            # ⭐ 修改8：过滤类别，只处理感兴趣的人车类别
            if int(cls) in class_id_to_name.keys():
                x_c, y_c, w, h = xyxy_to_xywh(*xyxy)
                xywh_bboxs.append([x_c, y_c, w, h])
                confs.append([conf.item()])
                oids.append(int(cls))

                detection_info.append({
                    'confidence': conf.item(),
                    'class': int(cls),
                    'bbox': [x.item() for x in xyxy],
                    'track_id': None
                })

        if len(xywh_bboxs) == 0:
            return ""

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            for i, track_id in enumerate(identities):
                if i < len(detection_info):
                    detection_info[i]['track_id'] = int(track_id)

            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities, detection_info)

        self.plotted_img = im0
        return ""


# ================= 主函数 =================
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict_div(cfg):
    global road_polygon

    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--polygon', nargs=8, type=int,
                        help='多边形坐标: x1 y1 x2 y2 x3 y3 x4 y4',
                        default=None)
    parser.add_argument('--draw', action='store_true',
                        help='是否手动绘制区域')
    args, unknown = parser.parse_known_args()

    init_tracker()
    cfg.model = cfg.model or "yolov8m.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.conf = 0.3

    # 获取道路区域
    if args.draw:
        # 手动绘制模式
        cap = cv2.VideoCapture(cfg.source)
        ret, first_frame = cap.read()
        cap.release()

        if ret:
            road_polygon = get_road_polygon_from_user(first_frame)
            print(f"最终多边形坐标: {road_polygon.tolist()}")
        else:
            print("无法读取视频第一帧，使用默认区域")
            road_polygon = np.array([[100, 400], [600, 300], [900, 600], [100, 500]], np.int32)

    elif args.polygon:
        # 命令行参数模式
        coords = args.polygon
        road_polygon = np.array([
            [coords[0], coords[1]],
            [coords[2], coords[3]],
            [coords[4], coords[5]],
            [coords[6], coords[7]]
        ], np.int32)
        print(f"使用命令行多边形坐标: {road_polygon.tolist()}")

    else:
        # 默认区域 - 覆盖更广的区域以同时捕捉人和车
        road_polygon = np.array([[0, 300], [1920, 200], [1920, 700], [0, 700]], np.int32)
        print(f"使用默认多边形坐标: {road_polygon.tolist()}")

    print(f"正在同时识别人和车辆...")
    print(f"目标类别: {', '.join(valid_classes)}")

    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict_div()