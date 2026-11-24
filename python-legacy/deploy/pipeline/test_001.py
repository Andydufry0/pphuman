import os
import cv2
import yaml
import argparse
from pipeline_02 import Pipeline, PipePredictor, get_model_dir

# ===================== 配置区域 =====================
ROOT_PATH = os.path.abspath(".")
CONFIG_SKELETON = os.path.join(ROOT_PATH, "config/infer_cfg_pphuman.yml")
TEST_IMG = r"E:\Image\Deep\01.jpg"
MODEL_ID = 1
# ===========================================================

def load_yml_config(yml_path):
    if not os.path.exists(yml_path):
        raise FileNotFoundError(f"配置文件不存在：{yml_path}")
    with open(yml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 官方模型自动下载处理
    cfg = get_model_dir(cfg)
    return cfg

def build_pipeline_args():
    # 仅占位args，所有推理参数不再依赖此处，不会报属性缺失
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./infer_output")
    parser.add_argument("--image_file", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--video_file", default=None)
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--camera_id", default=-1)
    parser.add_argument("--rtsp", nargs="+", default=None)
    args = parser.parse_args([])
    return args

if __name__ == "__main__":
    infer_cfg = load_yml_config(CONFIG_SKELETON)
    args = build_pipeline_args()

    pipe = Pipeline(args, infer_cfg)
    print("1. Pipeline初始化完成，未加载模型")

    det, track, kpt, action = pipe.load_target_model(model_id=MODEL_ID)
    print(f"2. {MODEL_ID}号模型加载完成")

    frame = cv2.imread(TEST_IMG)
    if frame is None:
        raise FileNotFoundError(f"图片读取失败，请检查路径：{TEST_IMG}")
    print(f"3. 读取图片成功，图像尺寸：{frame.shape}")

    # 执行单帧推理
    draw_img, res = pipe.run_single_frame(frame, model_id=MODEL_ID)
    print("=" * 50)
    print("推理输出结果：")
    print(res)
    print("=" * 50)

    # 展示结果窗口
    cv2.imshow("推理结果图", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存输出图片
    save_path = r"E:\Image\Deep\result.jpg"
    cv2.imwrite(save_path, draw_img)
    print(f"推理可视化图片已保存至 {save_path}")

    # 全部释放模型缓存
    pipe.unload_unused_models(keep_ids=[])
    print("4. 所有模型卸载完毕，显存释放完成")