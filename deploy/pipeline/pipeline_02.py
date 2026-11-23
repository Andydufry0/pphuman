# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import glob
import cv2
import numpy as np
import math
import paddle
import sys
import copy
import threading
import queue
import time

from collections import defaultdict
from datacollector import DataCollector, Result
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from cfg_utils import argsparser, print_arguments, merge_cfg
from pipe_utils import PipeTimer
from pipe_utils import get_test_images, crop_image_with_det, crop_image_with_mot, parse_mot_res, parse_mot_keypoint
from pipe_utils import PushStream

from python.infer import Detector, DetectorPicoDet
from python.keypoint_infer import KeyPointDetector
from python.keypoint_postprocess import translate_to_ori_images
from python.preprocess import decode_image, ShortSizeScale
# 官方原生pose可视化函数（真正官方绘制骨骼接口，优先使用）
from python.visualize import visualize_box_mask, visualize_attr, visualize_pose, visualize_action, visualize_vehicleplate, visualize_vehiclepress, visualize_lane, visualize_vehicle_retrograde

from pptracking.python.mot_sde_infer import SDE_Detector
from pptracking.python.mot.visualize import plot_tracking_dict
from pptracking.python.mot.utils import flow_statistic, update_object_info

from pphuman.attr_infer import AttrDetector
from pphuman.video_action_infer import VideoActionDetector
from pphuman.action_infer import SkeletonActionDetector, DetActionDetector, ClsActionDetector
from pphuman.action_utils import KeyPointBuff, ActionVisualHelper
from pphuman.reid_infer import ReidDetector
from pphuman.mtmct_infer import MTMCTDetector

from ppvehicle.vehicle_plate_infer import PlateDetector
from ppvehicle.vehicle_attr_infer import VehicleAttrDetector
from ppvehicle.vehicle_pressing_infer import PressingDetector
from ppvehicle.vehicle_retrograde_infer import RetrogradeDetector
from ppvehicle.lane_infer import LaneDetector

from download import auto_download_model

import threading


def get_model_dir(cfg):
    """模型路径预处理+自动下载权重，返回更新后的cfg"""
    for key in cfg.keys():
        if isinstance(cfg[key], dict) and ("enable" not in cfg[key] or cfg[key]['enable']):
            if "model_dir" in cfg[key]:
                model_dir = cfg[key]["model_dir"]
                downloaded_model_dir = auto_download_model(model_dir)
                if downloaded_model_dir:
                    model_dir = downloaded_model_dir
                    cfg[key]["model_dir"] = model_dir
            elif key == "VEHICLE_PLATE":
                det_model_dir = cfg[key]["det_model_dir"]
                downloaded_det_model_dir = auto_download_model(det_model_dir)
                if downloaded_det_model_dir:
                    cfg[key]["det_model_dir"] = downloaded_det_model_dir
                rec_model_dir = cfg[key]["rec_model_dir"]
                downloaded_rec_model_dir = auto_download_model(rec_model_dir)
                if downloaded_rec_model_dir:
                    cfg[key]["rec_model_dir"] = downloaded_rec_model_dir
        elif key == "MOT":
            model_dir = cfg[key]["model_dir"]
            downloaded_model_dir = auto_download_model(model_dir)
            if downloaded_model_dir:
                cfg[key]["model_dir"] = downloaded_model_dir
    return cfg


# ====================== 自定义绘图（仅测试备用，正式使用官方visualize_pose） ======================
# PP-Human 标准17关键点骨架
SKELETON = [
    [0, 1],   # 鼻-颈
    [1, 2],   # 颈-右肩
    [2, 3],   # 右肩-右肘
    [3, 4],   # 右肘-右手腕
    [1, 5],   # 颈-左肩
    [5, 6],   # 左肩-左肘
    [6, 7],   # 左肘-左手腕
    [1, 8],   # 颈-右髋
    [8, 9],   # 右髋-右膝
    [9, 10],  # 右膝-右脚踝
    [1, 11],  # 颈-左髋
    [11, 12], # 左髋-左膝
    [12, 13], # 左膝-左脚踝
    [0, 14],  # 鼻-右眼
    [0, 15],  # 鼻-左眼
    [14, 16]  # 右眼-右耳
]
POINT_COLOR = (0, 0, 255)    # 红色关节点
LINE_COLOR = (255, 0, 0)     # 蓝色骨骼连线
BOX_COLOR = (0, 255, 0)      # 绿色检测框


def draw_bbox(image, det_results, conf_thresh=0.5):
    img = image.copy()
    for res in det_results:
        score = res['score']
        if score < conf_thresh:
            continue
        x1, y1 = int(res['x1']), int(res['y1'])
        x2, y2 = int(res['x2']), int(res['y2'])
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thickness=2)
        text = f"human {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)
    return img


def draw_kpt(image, kpt_results, conf_thresh=0.3):
    img = image.copy()
    for person_kpt in kpt_results:
        kpts = np.array(person_kpt)
        x = kpts[:, 0]
        y = kpts[:, 1]
        conf = kpts[:, 2]
        # 绘制关键点
        for i in range(17):
            if conf[i] > conf_thresh:
                cv2.circle(img, (int(x[i]), int(y[i])), 4, POINT_COLOR, -1)
        # 绘制骨架连线
        for (s, e) in SKELETON:
            if conf[s] > conf_thresh and conf[e] > conf_thresh:
                cv2.line(img, (int(x[s]), int(y[s])), (int(x[e]), int(y[e])), LINE_COLOR, thickness=2)
    return img


# ====================== Pipeline 主调度类（已全部修复报错） ======================
class Pipeline:
    def __init__(self, args, cfg):
        self.args = args
        self.raw_cfg = cfg
        self.model_cache = dict()       # 缓存检测/跟踪/关键点模型实例
        self.predictor_cache = None     # 全局单例推理器，仅初始化一次

    def _build_detector(self, model_id: int):
        cfg = self.raw_cfg
        det_cfg = cfg["DET"]
        # 直接读取yaml顶层推理参数，无INFER节点，兼容官方原始配置
        device = cfg.get("device", "GPU")
        run_mode = cfg.get("run_mode", "paddle")
        trt_min_shape = cfg.get("trt_min_shape", 1)
        trt_max_shape = cfg.get("trt_max_shape", 960)
        trt_opt_shape = cfg.get("trt_opt_shape", 640)
        trt_calib_mode = cfg.get("trt_calib_mode", False)
        cpu_threads = cfg.get("cpu_threads", 6)
        enable_mkldnn = cfg.get("enable_mkldnn", False)

        return Detector(
            model_dir=det_cfg["model_dir"],
            device=device,
            run_mode=run_mode,
            batch_size=det_cfg["batch_size"],
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn
        )

    def _build_tracker(self, model_id: int):
        # 跟踪占位函数，按需自行完善
        cfg = self.raw_cfg
        pass

    def _build_kpt_model(self, model_id: int):
        cfg = self.raw_cfg
        kpt_cfg = cfg["KPT"]
        # 读取yaml顶层推理参数，不依赖INFER分组、不依赖args
        device = cfg.get("device", "GPU")
        run_mode = cfg.get("run_mode", "paddle")
        trt_min_shape = cfg.get("trt_min_shape", 1)
        trt_max_shape = cfg.get("trt_max_shape", 960)
        trt_opt_shape = cfg.get("trt_opt_shape", 640)
        trt_calib_mode = cfg.get("trt_calib_mode", False)
        cpu_threads = cfg.get("cpu_threads", 6)
        enable_mkldnn = cfg.get("enable_mkldnn", False)

        return KeyPointDetector(
            model_dir=kpt_cfg["model_dir"],
            device=device,
            run_mode=run_mode,
            batch_size=kpt_cfg["batch_size"],
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            use_dark=False
        )

    def _build_action_model(self, model_id: int):
        # 动作识别占位函数，按需自行完善
        cfg = self.raw_cfg
        pass

    def load_target_model(self, model_id: int):
        # 缓存命中直接返回，不重复初始化模型
        if model_id in self.model_cache:
            return self.model_cache[model_id]

        print(f"首次加载模型组: {model_id}")
        det = self._build_detector(model_id)
        track = self._build_tracker(model_id)
        kpt = self._build_kpt_model(model_id)
        action = self._build_action_model(model_id)

        self.model_cache[model_id] = (det, track, kpt, action)
        print(f"{model_id}号模型加载完成")

        # 仅首次加载模型时创建一次PipePredictor，全局复用
        if self.predictor_cache is None:
            self.predictor_cache = PipePredictor(self.args, self.raw_cfg)
        return self.model_cache[model_id]

    def run_single_frame(self, frame, model_id: int):
        det, track, kpt, action = self.load_target_model(model_id)
        pred = self.predictor_cache
        draw_img, res = pred.infer(frame, det, track, kpt, action)
        return draw_img, res

    def unload_unused_models(self, keep_ids=[]):
        # 释放指定外所有模型缓存
        del_ids = [mid for mid in self.model_cache.keys() if mid not in keep_ids]
        for mid in del_ids:
            del self.model_cache[mid]
        self.predictor_cache = None
        print(f"已释放闲置模型，当前缓存剩余 {len(self.model_cache)} 组模型")


# ====================== 推理核心类 PipePredictor ======================
class PipePredictor:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        # 读取检测、关键点配置
        det_cfg = self.cfg.get("DET", {})
        kpt_cfg = self.cfg.get("KPT", {})
        self.det_conf = det_cfg.get("conf_thresh", 0.5)
        self.kpt_thresh = kpt_cfg.get("visual_thresh", 0.3)
        self.use_kpt = kpt_cfg.get("enable", False)

        # 推理硬件参数全部从yaml cfg读取，彻底脱离args依赖，解决device缺失报错
        self.device = self.cfg.get("device", "GPU")
        self.run_mode = self.cfg.get("run_mode", "paddle")
        self.trt_min_shape = self.cfg.get("trt_min_shape", 1)
        self.trt_max_shape = self.cfg.get("trt_max_shape", 960)
        self.trt_opt_shape = self.cfg.get("trt_opt_shape", 640)
        self.trt_calib_mode = self.cfg.get("trt_calib_mode", False)
        self.cpu_threads = self.cfg.get("cpu_threads", 6)
        self.enable_mkldnn = self.cfg.get("enable_mkldnn", False)

    def infer(self, img, det_model, track_model, kpt_model, action_model):
        """单图推理主逻辑"""
        ori_img = img.copy()
        final_result = dict()

        # 1. 人体检测
        det_res = det_model.predict_image([img])[0]
        final_result["det"] = det_res
        vis_img = visualize_box_mask(ori_img, det_res, self.det_conf)

        # 2. 关键点识别（开启才执行）
        if self.use_kpt and len(det_res['boxes']) > 0:
            kpt_input_imgs, bbox_offset = crop_image_with_det(img, det_res)
            kpt_output = kpt_model.predict_images(kpt_input_imgs)
            kpt_res = translate_to_ori_images(kpt_output, bbox_offset)
            final_result["keypoint"] = kpt_res
            # 官方骨骼绘制
            vis_img = visualize_pose(vis_img, kpt_res, visual_thresh=self.kpt_thresh)

        return vis_img, final_result