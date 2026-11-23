# Copyright (c) 2024 PaddleDetection Authors. All Rights Reserved.
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

"""
扩展行为检测模块
包含：人群聚集、徘徊滞留、奔跑异常速度、异常姿态
"""

import numpy as np
from collections import defaultdict, deque
import time


class CrowdGatheringDetector:
    """
    人群聚集检测
    基于 MOT 结果统计区域内人数，超过阈值则报警
    """
    
    def __init__(self, 
                 region=None,  # [x1, y1, x2, y2] 或 None 表示全图
                 crowd_threshold=5,  # 人数阈值
                 alert_duration=3.0):  # 持续时间（秒）
        self.region = region
        self.crowd_threshold = crowd_threshold
        self.alert_duration = alert_duration
        self.alert_start_time = None
        self.is_alerting = False
        self.was_alerting = False
        
    def update(self, mot_res, frame_width, frame_height, fps, frame_id=0):
        """
        更新检测结果
        
        Args:
            mot_res: MOT 检测结果，格式 {'boxes': np.array([[id, class, score, xmin, ymin, w, h], ...])}
            frame_width: 帧宽度
            frame_height: 帧高度
            fps: 帧率
            frame_id: 当前帧号（用于换算虚拟时间，避免受推理速度影响）
            
        Returns:
            dict: {'is_crowd': bool, 'person_count': int, 'alert': bool}
        """
        boxes = mot_res.get('boxes', [])
        
        if len(boxes) == 0:
            self.alert_start_time = None
            self.is_alerting = False
            self.was_alerting = False
            return {'is_crowd': False, 'person_count': 0, 'alert': False,
                    'alerting': False}
        
        # 确定检测区域
        if self.region is None:
            x1, y1, x2, y2 = 0, 0, frame_width, frame_height
        else:
            x1, y1, x2, y2 = self.region
        
        # 统计区域内人数（box: [id, class, score, xmin, ymin, w, h]）
        person_count = 0
        for box in boxes:
            person_center_x = box[3] + box[5] / 2
            person_center_y = box[4] + box[6] / 2
            
            if x1 <= person_center_x <= x2 and y1 <= person_center_y <= y2:
                person_count += 1
        
        # 判断是否聚集
        is_crowd = person_count >= self.crowd_threshold
        
        # 更新报警状态（用视频虚拟时间：帧号/fps）
        current_time = frame_id / max(fps, 1e-6)
        if is_crowd:
            if self.alert_start_time is None:
                self.alert_start_time = current_time
            elif current_time - self.alert_start_time >= self.alert_duration:
                self.is_alerting = True
        else:
            self.alert_start_time = None
            self.is_alerting = False
        
        # 只在状态上升沿返回报警，避免逐帧刷屏
        alert_rise = self.is_alerting and not self.was_alerting
        self.was_alerting = self.is_alerting

        return {
            'is_crowd': is_crowd,
            'person_count': person_count,
            'alert': alert_rise,
            'alerting': self.is_alerting
        }


class LoiteringDetector:
    """
    徘徊/滞留检测
    追踪每个人在特定区域的停留时间，超过阈值则报警
    """
    
    def __init__(self,
                 region=None,  # [x1, y1, x2, y2] 或 None 表示全图
                 loitering_threshold=10.0,  # 停留时间阈值（秒）
                 movement_threshold=50.0):  # 移动距离阈值（像素），小于此值视为滞留
        self.region = region
        self.loitering_threshold = loitering_threshold
        self.movement_threshold = movement_threshold
        
        # 追踪每个人的位置和停留时间
        self.person_tracker = defaultdict(lambda: {
            'first_seen': None,
            'first_position': None,
            'last_position': None,
            'total_movement': 0.0,
            'is_loitering': False
        })
        
    def update(self, mot_res, frame_width, frame_height, fps, frame_id=0):
        """
        更新检测结果
        
        Returns:
            dict: {track_id: {'is_loitering': bool, 'duration': float, 'movement': float}}
        """
        boxes = mot_res.get('boxes', [])
        # 用视频虚拟时间（帧号/fps），避免受推理速度影响
        current_time = frame_id / max(fps, 1e-6)
        
        # 确定检测区域
        if self.region is None:
            x1, y1, x2, y2 = 0, 0, frame_width, frame_height
        else:
            x1, y1, x2, y2 = self.region
        
        # 当前帧中检测到的人
        current_ids = set()
        
        loitering_results = {}
        
        for box in boxes:
            track_id = int(box[0])
            person_center_x = box[3] + box[5] / 2
            person_center_y = box[4] + box[6] / 2
            
            # 检查是否在区域内
            if not (x1 <= person_center_x <= x2 and y1 <= person_center_y <= y2):
                continue
            
            current_ids.add(track_id)
            tracker = self.person_tracker[track_id]
            
            # 首次出现
            if tracker['first_seen'] is None:
                tracker['first_seen'] = current_time
                tracker['first_position'] = (person_center_x, person_center_y)
                tracker['last_position'] = (person_center_x, person_center_y)
                tracker['total_movement'] = 0.0
            
            # 计算移动距离（累计路径，仅作参考）
            if tracker['last_position'] is not None:
                dx = person_center_x - tracker['last_position'][0]
                dy = person_center_y - tracker['last_position'][1]
                distance = np.sqrt(dx*dx + dy*dy)
                tracker['total_movement'] += distance
            
            tracker['last_position'] = (person_center_x, person_center_y)
            
            # 净位移：当前点相对首次出现点的距离（站着微动不会累计）
            fp = tracker['first_position']
            net_displacement = np.sqrt((person_center_x - fp[0])**2 +
                                       (person_center_y - fp[1])**2)
            
            # 计算停留时间
            duration = current_time - tracker['first_seen']
            
            # 判断是否徘徊（停留时间长且净位移小）
            is_loitering = (duration >= self.loitering_threshold and 
                          net_displacement < self.movement_threshold)
            
            tracker['is_loitering'] = is_loitering
            # 上升沿报警（只在刚判定徘徊的那一帧为 True）
            alert_rise = is_loitering and not tracker.get('was_loitering', False)
            tracker['was_loitering'] = is_loitering

            loitering_results[track_id] = {
                'is_loitering': is_loitering,
                'alert': alert_rise,
                'duration': duration,
                'movement': net_displacement
            }
        
        # 清理消失的追踪
        disappeared_ids = set(self.person_tracker.keys()) - current_ids
        for track_id in disappeared_ids:
            del self.person_tracker[track_id]
        
        return loitering_results


class RunningDetector:
    """
    奔跑/异常速度检测
    基于关键点或 bbox 中心点计算移动速度
    """
    
    def __init__(self,
                 speed_threshold=200.0,  # 速度阈值（像素/秒）
                 min_frames=5):  # 最少追踪帧数
        self.speed_threshold = speed_threshold
        self.min_frames = min_frames
        
        # 追踪每个人的位置和速度
        self.person_tracker = defaultdict(lambda: {
            'positions': deque(maxlen=10),  # 最近 10 个位置
            'timestamps': deque(maxlen=10),  # 对应时间戳
            'is_running': False
        })
        
    def update(self, mot_res, fps, frame_id=0):
        """
        更新检测结果
        
        Returns:
            dict: {track_id: {'is_running': bool, 'speed': float}}
        """
        boxes = mot_res.get('boxes', [])
        # 用视频虚拟时间（帧号/fps），保证速度单位是真实的像素/秒
        current_time = frame_id / max(fps, 1e-6)
        
        running_results = {}
        current_ids = set()
        
        for box in boxes:
            track_id = int(box[0])
            person_center_x = box[3] + box[5] / 2
            person_center_y = box[4] + box[6] / 2
            
            current_ids.add(track_id)
            tracker = self.person_tracker[track_id]
            
            # 记录位置和时间的
            tracker['positions'].append((person_center_x, person_center_y))
            tracker['timestamps'].append(current_time)
            
            # 计算速度（至少需要 2 个点）
            if len(tracker['positions']) >= 2:
                # 使用最近几个点计算平均速度
                positions = list(tracker['positions'])
                timestamps = list(tracker['timestamps'])
                
                # 计算总距离和总时间
                total_distance = 0.0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    total_distance += np.sqrt(dx*dx + dy*dy)
                
                total_time = timestamps[-1] - timestamps[0]
                
                if total_time > 0:
                    speed = total_distance / total_time  # 像素/秒
                    
                    # 判断是否奔跑
                    is_running = speed >= self.speed_threshold
                    tracker['is_running'] = is_running
                    # 上升沿报警
                    alert_rise = is_running and not tracker.get('was_running', False)
                    tracker['was_running'] = is_running

                    running_results[track_id] = {
                        'is_running': is_running,
                        'alert': alert_rise,
                        'speed': speed
                    }
        
        # 清理消失的追踪
        disappeared_ids = set(self.person_tracker.keys()) - current_ids
        for track_id in disappeared_ids:
            del self.person_tracker[track_id]
        
        return running_results


class AbnormalPostureDetector:
    """
    异常姿态检测
    基于关键点分析人体姿态，检测异常姿态（如倒地、扭曲等）
    """
    
    def __init__(self,
                 fall_angle_threshold=60.0,  # 倒地角度阈值（度）
                 symmetry_threshold=0.3):  # 对称性阈值
        self.fall_angle_threshold = fall_angle_threshold
        self.symmetry_threshold = symmetry_threshold
        self.posture_state = {}  # track_id -> 上次是否异常
        
    def analyze_posture(self, keypoints):
        """
        分析单个人的姿态
        
        Args:
            keypoints: 关键点数组，shape [17, 2] 或 [17, 3]（含置信度）
                      COCO 格式：0-鼻子, 5-左肩, 6-右肩, 11-左髋, 12-右髋, ...
                      
        Returns:
            dict: {'is_abnormal': bool, 'posture_type': str, 'confidence': float}
        """
        if keypoints is None or len(keypoints) < 17:
            return {'is_abnormal': False, 'posture_type': 'unknown', 'confidence': 0.0}
        
        # 转换为 numpy array
        keypoints = np.array(keypoints)
        
        # 提取关键点（COCO 格式）
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # 检查关键点置信度
        if keypoints.shape[1] == 3:
            confidences = keypoints[:, 2]
            if confidences[0] < 0.3 or confidences[5] < 0.3 or confidences[6] < 0.3:
                return {'is_abnormal': False, 'posture_type': 'low_confidence', 'confidence': 0.0}
        
        # 计算躯干角度（判断是否倒地）
        shoulder_center = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ])
        hip_center = np.array([
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        ])
        
        # 计算躯干向量
        torso_vector = shoulder_center - hip_center
        # 计算与垂直方向的夹角
        vertical_vector = np.array([0, -1])  # 向上
        cos_angle = np.dot(torso_vector, vertical_vector) / (
            np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector) + 1e-6
        )
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # 判断是否倒地
        is_fallen = angle > self.fall_angle_threshold
        
        # 计算身体对称性（判断是否扭曲）
        left_shoulder_hip_dist = np.linalg.norm(left_shoulder - left_hip)
        right_shoulder_hip_dist = np.linalg.norm(right_shoulder - right_hip)
        
        if left_shoulder_hip_dist > 0 and right_shoulder_hip_dist > 0:
            symmetry = abs(left_shoulder_hip_dist - right_shoulder_hip_dist) / max(
                left_shoulder_hip_dist, right_shoulder_hip_dist
            )
            is_twisted = symmetry > self.symmetry_threshold
        else:
            is_twisted = False
        
        # 综合判断
        is_abnormal = is_fallen or is_twisted
        
        if is_fallen:
            posture_type = 'fallen'
        elif is_twisted:
            posture_type = 'twisted'
        else:
            posture_type = 'normal'
        
        confidence = 1.0 - (angle / 180.0) if is_fallen else (1.0 - symmetry)
        
        return {
            'is_abnormal': is_abnormal,
            'posture_type': posture_type,
            'confidence': confidence,
            'angle': angle
        }
    
    def update(self, kpt_res, mot_res):
        """
        更新检测结果
        
        Args:
            kpt_res: 关键点检测结果
            mot_res: MOT 检测结果
            
        Returns:
            dict: {track_id: {'is_abnormal': bool, 'posture_type': str, 'confidence': float}}
        """
        keypoints_list = kpt_res.get('keypoint', [[]])[0]
        boxes = mot_res.get('boxes', [])
        
        abnormal_results = {}
        
        for idx, (kpts, box) in enumerate(zip(keypoints_list, boxes)):
            track_id = int(box[0])
            
            # 分析姿态
            posture_result = self.analyze_posture(kpts)
            # 上升沿报警
            is_abn = posture_result.get('is_abnormal', False)
            posture_result['alert'] = is_abn and not self.posture_state.get(
                track_id, False)
            self.posture_state[track_id] = is_abn
            
            abnormal_results[track_id] = posture_result
        
        return abnormal_results
