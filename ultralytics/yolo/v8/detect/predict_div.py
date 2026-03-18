import sys
import os
import time
from collections import deque
import numpy as np
import cv2
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# ===================== 全局变量 =====================

data_deque = {}

# 速度
object_speed = {}
object_last_pos = {}
object_last_time = {}

# 计数
line_y = 400
offset = 10

entered_ids = set()
exited_ids = set()

enter_count = {"car":0, "truck":0, "bus":0}
exit_count = {"car":0, "truck":0, "bus":0}

# ===================== DeepSORT =====================
deepsort = None

def init_tracker():
    global deepsort
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )

# ===================== 工具函数 =====================

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = bbox_left + bbox_w / 2
    y_c = bbox_top + bbox_h / 2
    return x_c, y_c, bbox_w, bbox_h

def compute_color_for_labels(label):
    if label == 2:
        return (222,82,175)
    elif label == 7:
        return (0,255,255)
    elif label == 5:
        return (0,149,255)
    return (255,255,255)

# ===================== 核心：画框 + 速度 + 计数 =====================

def draw_boxes(img, bbox, names, object_id, identities):

    global object_speed, object_last_pos, object_last_time
    global enter_count, exit_count, entered_ids, exited_ids

    h, w, _ = img.shape

    # 清理丢失目标
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):

        x1, y1, x2, y2 = map(int, box)
        center = (int((x1+x2)/2), int((y2+y2)/2))

        id = int(identities[i])
        cls_id = int(object_id[i])
        obj_name = names[cls_id]

        # 只统计三类
        if obj_name not in ["car","truck","bus"]:
            continue

        # 轨迹缓存
        if id not in data_deque:
            data_deque[id] = deque(maxlen=30)

        data_deque[id].appendleft(center)

        # ================= 速度 =================
        now = time.time()

        if id in object_last_pos:
            prev = object_last_pos[id]
            dist = np.linalg.norm(np.array(center) - np.array(prev))
            dt = now - object_last_time[id]

            if dt > 0:
                speed = dist / dt
                speed_kmh = speed * 0.05  # ⚠️需调参
                object_speed[id] = int(speed_kmh)

        object_last_pos[id] = center
        object_last_time[id] = now

        speed = object_speed.get(id, 0)

        # ================= 过线计数 =================
        if len(data_deque[id]) >= 2:
            prev_center = data_deque[id][1]

            if prev_center[1] < line_y and center[1] >= line_y:
                if id not in entered_ids:
                    entered_ids.add(id)
                    enter_count[obj_name] += 1

            elif prev_center[1] > line_y and center[1] <= line_y:
                if id not in exited_ids:
                    exited_ids.add(id)
                    exit_count[obj_name] += 1

        # ================= 画框 =================
        color = compute_color_for_labels(cls_id)
        label = f"{id}:{obj_name} {speed}km/h"

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, label, (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 轨迹线
        for j in range(1, len(data_deque[id])):
            if data_deque[id][j-1] is None or data_deque[id][j] is None:
                continue
            cv2.line(img, data_deque[id][j-1],
                     data_deque[id][j], color, 2)

    # ================= 画分界线 =================
    cv2.line(img, (0,line_y), (w,line_y), (0,255,0), 3)

    # ================= UI =================

    # 左：离开
    cv2.putText(img, "Leaving", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    y = 80
    for k,v in exit_count.items():
        cv2.putText(img, f"{k}:{v}", (20,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        y += 30

    # 右：进入
    cv2.putText(img, "Entering", (w-200,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    y = 80
    for k,v in enter_count.items():
        cv2.putText(img, f"{k}:{v}", (w-200,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        y += 30

    return img

# ===================== Predictor =====================

class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.float() / 255.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou)
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch

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

            draw_boxes(im0, bbox_xyxy, self.model.names,
                       object_id, identities)

        return ""

# ===================== 启动 =====================

import hydra

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()