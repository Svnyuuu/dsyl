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

# 速度
speed_dict = {}

# 参数（需要你调）
fps = 30
ppm = 8   # pixel per meter

line_y = 0

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
    if label == 2:   # car
        return (222, 82, 175)
    elif label == 5: # bus
        return (0, 149, 255)
    elif label == 7: # truck
        return (255, 204, 0)
    else:
        return (255, 255, 255)


def UI_box(x, img, color=None, label=None, line_thickness=2):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness)

    if label:
        cv2.putText(img, label, (c1[0], c1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ================= 核心：画框 + 计数 + 速度 =================
def draw_boxes(img, bbox, names, object_id, identities=None):
    global line_y

    h, w, _ = img.shape
    line_y = int(h * 0.6)

    # 画分界线
    cv2.line(img, (0, line_y), (w, line_y), (0, 255, 0), 2)

    # 清理丢失ID
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = map(int, box)

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        track_id = int(identities[i])
        cls_id = int(object_id[i])
        obj_name = names[cls_id]

        if obj_name not in ["car", "truck", "bus"]:
            continue

        if track_id not in data_deque:
            data_deque[track_id] = deque(maxlen=30)

        data_deque[track_id].appendleft(center)

        # ===== 速度 =====
        speed = 0
        if len(data_deque[track_id]) >= 2:
            p1 = data_deque[track_id][0]
            p2 = data_deque[track_id][1]
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            speed = (dist * fps) / ppm * 3.6
            speed_dict[track_id] = speed

        # ===== 过线判断 =====
        if len(data_deque[track_id]) >= 2:
            prev_y = data_deque[track_id][1][1]
            curr_y = data_deque[track_id][0][1]

            if track_id not in counted_ids:
                # 进入
                if prev_y < line_y and curr_y >= line_y:
                    enter_count[obj_name] += 1
                    counted_ids.add(track_id)

                # 离开
                elif prev_y > line_y and curr_y <= line_y:
                    leave_count[obj_name] += 1
                    counted_ids.add(track_id)

        # ===== 画框 =====
        color = compute_color_for_labels(cls_id)
        label = f"{track_id}:{obj_name} {int(speed)}km/h"
        UI_box(box, img, color, label)

        # ===== 轨迹 =====
        for j in range(1, len(data_deque[track_id])):
            cv2.line(img,
                     data_deque[track_id][j - 1],
                     data_deque[track_id][j],
                     color, 2)

    # ===== 显示统计 =====
    y = 40
    for k, v in leave_count.items():
        cv2.putText(img, f"Leaving {k}:{v}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 30

    y = 40
    for k, v in enter_count.items():
        cv2.putText(img, f"Entering {k}:{v}", (w - 300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 30

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
        #self.annotator = self.get_annotator(im0)

        det = preds[idx]
        if len(det) == 0:
            return ""

        xywh_bboxs, confs, oids = [], [], []

        for *xyxy, conf, cls in det:
            x_c, y_c, w, h = xyxy_to_xywh(*xyxy)
            xywh_bboxs.append([x_c, y_c, w, h])
            confs.append([conf.item()])
            oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

        self.plotted_img = im0
        return ""


# ================= 主函数 =================
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict_div(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8m.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)

    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict_div()