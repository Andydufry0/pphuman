# PP-Human 行为监测

基于 [PaddleDetection 2.9](https://github.com/PaddlePaddle/PaddleDetection) 的 PP-Human 行人行为分析部署代码，支持行为识别：

- 摔倒（FALL，STGCN 骨架时序）
- 打架（FIGHT，ppTSM 视频分类）
- 抽烟（SMOKE，ID-based 目标检测）
- 打电话（CALL，ID-based 图像分类）
- 人群聚集 / 徘徊 / 奔跑 / 异常姿态（规则）

本仓库包含两套实现，共用同一套 PP-Human 模型：

| 目录 | 技术栈 | 用途 |
| --- | --- | --- |
| [`qt/`](qt/) | **C++17 / Qt5 / TensorRT 10** | 多路实时运行、工业报警、相机 / RTSP / HLS 接入 |
| [`deploy/`](deploy/) [`configs/`](configs/) | Python / PaddlePaddle | 模型转换、算法对照、离线验证 |

C++ 版说明见 [`qt/README.md`](qt/README.md)。

## 目录结构

```
deploy/pipeline/           # PP-Human 主流程
  pipeline.py              # 入口（支持 视频 / 图片 / RTSP）
  pphuman/                 # 行为模型封装
    action_infer.py        # 摔倒/打架/抽烟/打电话推理
    action_utils.py        # 关键点缓冲、结果可视化
    extended_actions.py    # 聚集/徘徊/奔跑/姿态规则
    scheduler.py           # 时间门控（分频推理）
    trt_backend.py         # TensorRT 后端
deploy/python/             # 检测/关键点/预处理/可视化基础模块
deploy/pptracking/         # 多目标跟踪（ByteTrack / SDE）
configs/pphuman/           # 各模型训练配置
config/infer_cfg_pphuman.yml  # 推理配置（模型路径、阈值、开关）
```

## 使用方式

本仓库只包含**部署推理部分**，需置于完整 PaddleDetection 2.9 根目录下使用。

```bash
# 1. 下载推理模型（放到 infer_cfg_pphuman.yml 里配置的 model_dir）
python deploy/pipeline/download.py   # 或按官方文档手动下载

# 2. 跑单视频
python deploy/pipeline/pipeline.py \
  --config deploy/pipeline/config/infer_cfg_pphuman.yml \
  --video_file=test.mp4 \
  --device=GPU --run_mode=paddle

# 3. 跑 RTSP 流
python deploy/pipeline/pipeline.py \
  --config deploy/pipeline/config/infer_cfg_pphuman.yml \
  --rtsp=rtsp://user:pass@ip:554/stream \
  --device=GPU
```

## 关键参数（config/infer_cfg_pphuman.yml）

| 模块 | 参数 | 官方默认 |
|---|---|---|
| SKELETON_ACTION (摔倒) | max_frames / coord_size | 50 / [384, 512] |
| ID_BASED_DETACTION (抽烟) | threshold / interval_ms / display_frames | 0.65 / 500 / 20 |
| VIDEO_ACTION (打架) | frame_len / sample_freq | 8 / 7 |
| SPEED / 跟踪 | track_high / low / new / buffer | 0.3 / 0.2 / 0.4 / 30 |

## 说明

- 摔倒（STGCN）对"蹲下/弯腰"存在固有误判，属于模型能力边界；生产环境建议叠加多帧确认或二次判据。
- 抽烟/打架/姿态这类模型，官方原始用途是**可视化**而非告警；若要用于报警，务必加多帧确认，否则单帧噪声会大量误报。

## 许可

代码源自 PaddleDetection（Apache-2.0），遵循同一许可。
