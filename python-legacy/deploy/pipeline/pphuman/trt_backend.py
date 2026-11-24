# -*- coding: utf-8 -*-
"""PP-Human TRT FP16 推理后端（原生 TensorRT API）

用 TensorRT 直接加载 trtexec 编好的 FP16 engine，替换三个重模型的 paddle 前向：
  - mot      : SDE_Detector          (deploy/pptracking/python/mot_sde_infer.py)
  - kpt      : KeyPointDetector      (deploy/python/keypoint_infer.py)
  - smoking  : Detector              (deploy/python/infer.py)

子类只覆盖 preprocess(缓存 inputs) 与 predict(引擎前向)，
preprocess ops / postprocess / 跟踪器 / 全部上层逻辑复用原实现。
"""
import os

import numpy as np

TRT_BIN = r"E:\Deep\TensorRT\TensorRT-10.16.1.11\bin"
TRT_ROOT = r"E:\pphuman\trt"

ENGINE = {
    "mot": os.path.join(TRT_ROOT, "mot_fp16.engine"),
    "kpt": os.path.join(TRT_ROOT, "kpt_fp16.engine"),
    "smoking": os.path.join(TRT_ROOT, "smoking_fp16.engine"),
    "smoking_batch": os.path.join(TRT_ROOT, "smoking_fp16_batch5.engine"),
}

# NMS 输出框数动态，按容量上限分配，执行后按 boxes_num 裁剪
OUT_CAP = 3000


class TrtRunner(object):
    """原生 TRT engine 推理，接口仿 ort_sess.run(None, feeds) -> list[np]。"""

    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self._trt, self._cudart = trt, cudart
        if TRT_BIN not in os.environ.get("PATH", ""):
            os.environ["PATH"] = TRT_BIN + ";" + os.environ["PATH"]
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(engine_path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        assert self.engine is not None, "engine load failed: " + engine_path
        self.ctx = self.engine.create_execution_context()
        self._init_cuda()
        self.in_names, self.out_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_names.append(name)
            else:
                self.out_names.append(name)
        self._buf = {}
        self._out_specs = None

    def _check(self, res):
        r = res[0] if isinstance(res, tuple) else res
        assert r is None or getattr(r, "value", 0) == 0, res

    def _init_cuda(self):
        c = self._cudart
        self._check(c.cudaSetDevice(0))
        self._check(c.cudaFree(0))
        err, self.stream = c.cudaStreamCreate()
        self._check(err)

    def _dev_buf(self, key, nbytes):
        c = self._cudart
        old = self._buf.get(key)
        if old is None or old[1] < nbytes:
            if old is not None:
                self._check(c.cudaFree(old[0]))
            err, dev = c.cudaMalloc(nbytes)
            self._check(err)
            self._buf[key] = (dev, nbytes)
        return self._buf[key][0]

    def run(self, feeds):
        """feeds: {input_name: np.float32}; 返回输出 numpy 列表（动态维按 OUT_CAP）。"""
        trt, c = self._trt, self._cudart
        for name in self.in_names:
            arr = np.ascontiguousarray(feeds[name], dtype=np.float32)
            static_shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            if any(d < 0 for d in static_shape) or static_shape != arr.shape:
                self.ctx.set_input_shape(name, arr.shape)
            dev = self._dev_buf("in_" + name, arr.nbytes)
            self._check(c.cudaMemcpy(dev, arr.ctypes.data, arr.nbytes,
                                     c.cudaMemcpyKind.cudaMemcpyHostToDevice))
            self.ctx.set_tensor_address(name, int(dev))

        out_specs = []
        for name in self.out_names:
            shape = tuple(int(d) for d in self.ctx.get_tensor_shape(name))
            cap_shape = tuple(d if d > 0 else OUT_CAP for d in shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(cap_shape) * np.dtype(dtype).itemsize)
            dev = self._dev_buf("out_" + name, nbytes)
            self.ctx.set_tensor_address(name, int(dev))
            out_specs.append((name, cap_shape, dtype, nbytes))

        self.ctx.execute_async_v3(stream_handle=int(self.stream))
        self._check(c.cudaStreamSynchronize(self.stream))

        results = []
        for name, cap_shape, dtype, nbytes in out_specs:
            out = np.empty(cap_shape, dtype=dtype)
            dev = self._dev_buf("out_" + name, nbytes)
            self._check(c.cudaMemcpy(out.ctypes.data, dev, nbytes,
                                     c.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            results.append(out)
        self._out_specs = out_specs
        return results

    def run_dev(self, dev_feeds, out_caps=None):
        """用已驻显存的输入直接执行（跳过 H2D）。dev_feeds: {name: (dev_ptr, shape)}。"""
        trt, c = self._trt, self._cudart
        for name in self.in_names:
            dev, shape = dev_feeds[name]
            self.ctx.set_input_shape(name, tuple(int(d) for d in shape))
            self.ctx.set_tensor_address(name, int(dev))
        out_specs = []
        for name in self.out_names:
            shape = tuple(int(d) for d in self.ctx.get_tensor_shape(name))
            cap = tuple(d if d > 0 else (out_caps.get(name, 1) if out_caps else 1)
                        for d in shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(cap) * np.dtype(dtype).itemsize)
            dev = self._dev_buf("out_" + name, nbytes)
            self.ctx.set_tensor_address(name, int(dev))
            out_specs.append((name, cap, dtype, nbytes))
        self.ctx.execute_async_v3(stream_handle=int(self.stream))
        self._check(c.cudaStreamSynchronize(self.stream))
        results = []
        for name, cap_shape, dtype, nbytes in out_specs:
            out = np.empty(cap_shape, dtype=dtype)
            dev = self._dev_buf("out_" + name, nbytes)
            self._check(c.cudaMemcpy(out.ctypes.data, dev, nbytes,
                                     c.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            results.append(out)
        return results


def trim_by_num(boxes, num):
    """按 boxes_num 裁剪 NMS 输出（boxes: [CAP,6] -> [N,6]）。"""
    n = int(np.asarray(num).reshape(-1)[0]) if np.asarray(num).size else 0
    return boxes[:n]


# ---------------------------------------------------------------------------
# 批量抽烟路径：CPU 只做 resize(uint8)，归一化/HWC->CHW 在 GPU kernel 完成，
# 引擎为去 NMS 动态 batch 版，NMS 用 numpy 按人做。
# ---------------------------------------------------------------------------
_NORM_SRC = r"""
extern "C" __global__ void u8_to_norm(const unsigned char* __restrict__ src,
                                      float* __restrict__ dst,
                                      int n_imgs, int h, int w) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n_imgs * h * w;
    if (idx >= total) return;
    int n = (int)(idx / ((long)h * w));
    int rem = (int)(idx - (long)n * h * w);
    int y = rem / w, x = rem % w;
    const unsigned char* p = src + ((long)n * h + y) * w * 3 + x * 3;
    // crops 为 BGR 直通（与 paddle decode_image 行为一致），通道0/1/2 = B/G/R
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float inv_std[3] = {1.f/0.229f, 1.f/0.224f, 1.f/0.225f};
    for (int c = 0; c < 3; ++c) {
        float v = (float)p[c] / 255.f;
        dst[((long)n * 3 + c) * h * w + y * w + x] = (v - mean[c]) * inv_std[c];
    }
}
"""


def numpy_nms(boxes_xyxy, scores, iou_thr=0.6, score_thr=0.3, pre_topk=500, max_out=100):
    """单图 NMS（对齐 paddle MulticlassNMS3 语义，score_thr 提高到 0.3：
    上层判据 threshold=0.65，NMS 只压制更低分框，不影响任何 >0.65 的检出）。
    返回 [label, score, x1, y1, x2, y2] float32（label 恒 0）。"""
    keep = scores > score_thr
    b = boxes_xyxy[keep]
    s = scores[keep]
    if s.size == 0:
        return np.zeros((0, 6), np.float32)
    order = s.argsort()[::-1][:pre_topk]
    b, s = b[order], s[order]
    order = np.arange(s.size)
    sel = []
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    while order.size and len(sel) < max_out:
        i = order[0]
        sel.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0])
        yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2])
        yy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    b, s = b[sel], s[sel]
    n = len(sel)
    out = np.zeros((n, 6), np.float32)
    out[:, 1] = s
    out[:, 2:6] = b
    return out


class BatchSmokeRunner(object):
    """uint8 (N,640,640,3) -> GPU 归一化 -> 动态 batch 去 NMS 引擎 -> numpy NMS。
    predictimages 直接给每人框列表，绕开 paddle 逐人 preprocess。"""

    def __init__(self, engine_path=None):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        from cuda.core import Device, Program, ProgramOptions
        if TRT_BIN not in os.environ.get("PATH", ""):
            os.environ["PATH"] = TRT_BIN + ";" + os.environ["PATH"]
        self._trt, self._cudart = trt, cudart
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(engine_path or ENGINE["smoking_batch"], "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        assert self.engine is not None, "smoking_batch engine load failed"
        self.ctx = self.engine.create_execution_context()
        c = self._cudart
        self._check(c.cudaSetDevice(0))
        self._check(c.cudaFree(0))
        self._buf = {}
        self.in_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                         if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.out_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                          if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        # 统一使用 cuda.core 流（copy / kernel / 引擎全走它，避免多流竞态）
        self._dev = Device(0)
        self._dev.set_current()
        self._core_stream = self._dev.create_stream()
        self.stream = int(self._core_stream.__cuda_stream__()[0])
        prog = Program(_NORM_SRC, code_type="c++", options=ProgramOptions())
        self._ker = prog.compile("cubin").get_kernel("u8_to_norm")
        # scale_factor 常驻 device（全 1，最多 64 人）
        sf = np.ones((64, 2), np.float32)
        self._dev_sf = self._buf_alloc("sf", sf.nbytes)
        self._check(c.cudaMemcpy(self._dev_sf, sf.ctypes.data, sf.nbytes,
                                 c.cudaMemcpyKind.cudaMemcpyHostToDevice))

    def _check(self, res):
        r = res[0] if isinstance(res, tuple) else res
        assert r is None or getattr(r, "value", 0) == 0, res

    def _buf_alloc(self, key, nbytes):
        c = self._cudart
        old = self._buf.get(key)
        if old is None or old[1] < nbytes:
            if old is not None:
                self._check(c.cudaFree(old[0]))
            err, dev = c.cudaMalloc(nbytes)
            self._check(err)
            self._buf[key] = (dev, nbytes)
        return self._buf[key][0]

    def run_raw(self, u8_stack, sf=None):
        """u8_stack: (N,640,640,3) uint8 contiguous。
        sf: (N,2) 缩放比 [640/h, 640/w]（输出坐标 = 640 空间 / sf = 原 crop 坐标）。
        返回 (boxes[N,8400,4], scores[N,1,8400])。"""
        trt, c = self._trt, self._cudart
        from cuda.core import LaunchConfig, launch
        n = u8_stack.shape[0]
        h, w = u8_stack.shape[1], u8_stack.shape[2]
        dev_u8 = self._buf_alloc("u8", u8_stack.nbytes)
        self._check(c.cudaMemcpy(dev_u8, u8_stack.ctypes.data, u8_stack.nbytes,
                                 c.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if sf is None:
            sf = np.ones((n, 2), np.float32)
        sf = np.ascontiguousarray(sf, np.float32)
        dev_sf = self._buf_alloc("sf", sf.nbytes)
        self._check(c.cudaMemcpy(dev_sf, sf.ctypes.data, sf.nbytes,
                                 c.cudaMemcpyKind.cudaMemcpyHostToDevice))
        dev_norm = self._buf_alloc("norm", n * 3 * h * w * 4)
        lc = LaunchConfig(grid=(int((n * h * w + 255) // 256),), block=(256,))
        launch(self._core_stream, lc, self._ker,
               int(dev_u8), int(dev_norm), np.int32(n), np.int32(h), np.int32(w))
        # 引擎输入输出
        for name in self.in_names:
            if name == "scale_factor":
                arr_dev, arr_shape = dev_sf, (n, 2)
            else:
                arr_dev, arr_shape = dev_norm, (n, 3, h, w)
            self.ctx.set_input_shape(name, arr_shape)
            self.ctx.set_tensor_address(name, int(arr_dev))
        out_specs = []
        for name in self.out_names:
            shape = tuple(int(d) for d in self.ctx.get_tensor_shape(name))
            cap = tuple(d if d > 0 else 8400 for d in shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(cap) * np.dtype(dtype).itemsize)
            dev = self._buf_alloc("out_" + name, nbytes)
            self.ctx.set_tensor_address(name, int(dev))
            out_specs.append((name, cap, dtype, nbytes, dev))
        self.ctx.execute_async_v3(stream_handle=int(self.stream))
        self._check(c.cudaStreamSynchronize(self.stream))
        outs = []
        for name, cap, dtype, nbytes, dev in out_specs:
            arr = np.empty(cap, dtype=dtype)
            self._check(c.cudaMemcpy(arr.ctypes.data, dev, nbytes,
                                     c.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            outs.append(arr)
        return outs[0], outs[1]


def make_trt_det_batch(model_dir, device, run_mode, batch_size, args, **kwargs):
    """抽烟检测批量版：predict_image 一次吃全部 crop，GPU 归一化 + 批量引擎 + numpy NMS。"""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    runner = BatchSmokeRunner()
    pool = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 8)))
    cap_batch = 64
    r = cv2.resize  # 局部引用

    class BatchSmokeDetector(object):
        threshold = 0.65

        def predict_image(self, images, visual=False, *a, **k):
            N = len(images)
            if N == 0:
                return dict(boxes=np.zeros((0, 6), np.float32),
                            boxes_num=np.array([0]))
            # 线程池并行 resize（cv2 释放 GIL），直接写入预分配缓冲，免 stack 拷贝
            u8 = np.empty((N, 640, 640, 3), np.uint8)
            list(pool.map(lambda t: u8.__setitem__(
                t[0], r(t[1], (640, 640), interpolation=cv2.INTER_CUBIC)),
                enumerate(images)))
            # scale_factor = 640/原高宽，引擎内部用于把坐标除回原 crop 尺度
            sf = np.array([[640.0 / im.shape[0], 640.0 / im.shape[1]] for im in images],
                          np.float32)
            boxes_all = []
            num_list = []
            for st in range(0, N, cap_batch):
                chunk = np.ascontiguousarray(u8[st:st + cap_batch])
                bb, ss = runner.run_raw(chunk, sf[st:st + cap_batch])
                m = chunk.shape[0]
                for i in range(m):
                    boxes = numpy_nms(bb[i], ss[i, 0])
                    boxes_all.append(boxes)
                    num_list.append(len(boxes))
            if boxes_all:
                boxes_cat = np.concatenate(boxes_all, 0) if any(len(b) for b in boxes_all) \
                    else np.zeros((0, 6), np.float32)
            else:
                boxes_cat = np.zeros((0, 6), np.float32)
            return dict(boxes=boxes_cat, boxes_num=np.array(num_list))

    return BatchSmokeDetector()


class _TRTMixin(object):
    """缓存 preprocess 产物并用 TRT engine 执行前向。"""

    def _trt_init(self, name):
        self.trt_runner = TrtRunner(ENGINE[name])
        self.trt_in_names = self.trt_runner.in_names

    def preprocess(self, image_list):
        inputs = super().preprocess(image_list)
        self._trt_inputs = inputs
        return inputs

    def _trt_run(self, feeds):
        return self.trt_runner.run(feeds)


def make_trt_det(model_dir, device, run_mode, batch_size, args, trt_name):
    """python infer.Detector 的 TRT 版（抽烟检测用），batch_size=1 逐人跑。"""
    from python.infer import Detector

    class TRTDetector(_TRTMixin, Detector):
        def __init__(self):
            super().__init__(
                model_dir,
                device=device,
                run_mode=run_mode,
                batch_size=batch_size,
                trt_min_shape=args.trt_min_shape,
                trt_max_shape=args.trt_max_shape,
                trt_opt_shape=args.trt_opt_shape,
                trt_calib_mode=args.trt_calib_mode,
                cpu_threads=args.cpu_threads,
                enable_mkldnn=args.enable_mkldnn)
            self._trt_init(trt_name)

        def predict(self, repeats=1, run_benchmark=False):
            outs = self._trt_run({k: self._trt_inputs[k]
                                  for k in self.trt_in_names})
            boxes = trim_by_num(outs[0], outs[1])
            return dict(boxes=boxes, masks=None,
                        boxes_num=np.array([len(boxes)]))

    return TRTDetector()


def make_trt_kpt(model_dir, device, run_mode, batch_size, args, use_dark=False):
    """KeyPointDetector 的 TRT 版。"""
    from python.keypoint_infer import KeyPointDetector

    class TRTKeyPointDetector(_TRTMixin, KeyPointDetector):
        def __init__(self):
            super().__init__(
                model_dir,
                device=device,
                run_mode=run_mode,
                batch_size=batch_size,
                trt_min_shape=args.trt_min_shape,
                trt_max_shape=args.trt_max_shape,
                trt_opt_shape=args.trt_opt_shape,
                trt_calib_mode=args.trt_calib_mode,
                cpu_threads=args.cpu_threads,
                enable_mkldnn=args.enable_mkldnn,
                use_dark=use_dark)
            self._trt_init("kpt")

        def predict(self, repeats=1):
            outs = self._trt_run({"image": self._trt_inputs["image"]})
            return dict(heatmap=outs[0], masks=None)

    return TRTKeyPointDetector()


def make_trt_sde(model_dir, tracker_config, device, run_mode, batch_size,
                 args, **kwargs):
    """SDE_Detector 的 TRT 版（MOT 检测前向走 TRT，跟踪逻辑不变）。"""
    from pptracking.python.mot_sde_infer import SDE_Detector

    class _TRTSDE(_TRTMixin, SDE_Detector):
        def __init__(self):
            super().__init__(
                model_dir,
                tracker_config,
                device=device,
                run_mode=run_mode,
                batch_size=batch_size,
                trt_min_shape=args.trt_min_shape,
                trt_max_shape=args.trt_max_shape,
                trt_opt_shape=args.trt_opt_shape,
                trt_calib_mode=args.trt_calib_mode,
                cpu_threads=args.cpu_threads,
                enable_mkldnn=args.enable_mkldnn,
                **kwargs)
            self._trt_init("mot")

        def predict(self, repeats=1):
            outs = self._trt_run({k: self._trt_inputs[k]
                                  for k in self.trt_in_names})
            boxes = trim_by_num(outs[0], outs[1])
            return dict(boxes=boxes, boxes_num=np.array([len(boxes)]))

    return _TRTSDE()


# ---------------------------------------------------------------------------
# KPT 全 GPU 路径：整帧一次上卡，kernel 直接做 expand-crop + TopDownEvalAffine
# （各向同性缩放 s=192/crop_w）+ HWC->CHW，输出直接作为引擎输入，免 H2D 与 CPU 预处理。
# ---------------------------------------------------------------------------
_KPT_SRC = r"""
__device__ __forceinline__ float sample_u8(const unsigned char* f, int fh, int fw,
                                           int x, int y, int c) {
    if (x < 0 || y < 0 || x >= fw || y >= fh) return 0.f;
    return (float)f[((long)y * fw + x) * 3 + c];
}

extern "C" __global__ void kpt_affine(const unsigned char* __restrict__ frame,
                                      float* __restrict__ dst,
                                      const float* __restrict__ prm,
                                      int n, int oh, int ow, int fh, int fw) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n * oh * ow;
    if (idx >= total) return;
    int p = (int)(idx / ((long)oh * ow));
    int rem = (int)(idx - (long)p * oh * ow);
    int v = rem / ow, u = rem % ow;
    const float* q = prm + p * 7;
    float cx0 = q[0], cy0 = q[1], cx1 = q[2], cy1 = q[3];
    float s = q[4];
    float sx = (u - q[5]) / s + cx0;
    float sy = (v - q[6]) / s + cy0;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0;
    int ix1 = (int)(cx1 - 0.5f), iy1 = (int)(cy1 - 0.5f);
    for (int c = 0; c < 3; ++c) {
        // 裁剪框外补 0，与 cv2.warpAffine(BORDER_CONSTANT) 一致
        float v00 = (x0 >= (int)cx0 && x0 <= ix1 && y0 >= (int)cy0 && y0 <= iy1) ? sample_u8(frame, fh, fw, x0, y0, c) : 0.f;
        float v01 = (x0 + 1 >= (int)cx0 && x0 + 1 <= ix1 && y0 >= (int)cy0 && y0 <= iy1) ? sample_u8(frame, fh, fw, x0 + 1, y0, c) : 0.f;
        float v10 = (x0 >= (int)cx0 && x0 <= ix1 && y0 + 1 >= (int)cy0 && y0 + 1 <= iy1) ? sample_u8(frame, fh, fw, x0, y0 + 1, c) : 0.f;
        float v11 = (x0 + 1 >= (int)cx0 && x0 + 1 <= ix1 && y0 + 1 >= (int)cy0 && y0 + 1 <= iy1) ? sample_u8(frame, fh, fw, x0 + 1, y0 + 1, c) : 0.f;
        float top = v00 + (v01 - v00) * fx;
        float bot = v10 + (v11 - v10) * fx;
        dst[((long)p * 3 + c) * oh * ow + (long)v * ow + u] = top + (bot - top) * fy;
    }
}
"""

KPT_IN_W, KPT_IN_H = 192, 256


class GpuKptCropper(object):
    """整帧 uint8 -> (n,3,256,192) float32 CHW（GPU），数值等价官方 TopDownEvalAffine。"""

    def __init__(self):
        from cuda.core import Device, Program, ProgramOptions
        self._dev = Device(0)
        self._dev.set_current()
        self._core_stream = self._dev.create_stream()
        from cuda.bindings import runtime as cudart
        self._c = cudart
        self._c.cudaSetDevice(0)
        self._c.cudaFree(0)
        prog = Program(_KPT_SRC, code_type="c++", options=ProgramOptions())
        self._ker = prog.compile("cubin").get_kernel("kpt_affine")
        self._buf = {}

    def _alloc(self, key, nbytes):
        old = self._buf.get(key)
        if old is None or old[1] < nbytes:
            if old is not None:
                self._c.cudaFree(old[0])
            _, dev = self._c.cudaMalloc(nbytes)
            self._buf[key] = (dev, nbytes)
        return self._buf[key][0]

    def build(self, frame, rects):
        """frame: HxWx3 uint8 BGR；rects: [(x0,y0,x1,y1)] 已 expand+clip 的裁剪框。
        返回 (dev_ptr, (n,3,256,192), params)。"""
        from cuda.core import LaunchConfig, launch
        n = len(rects)
        fh, fw = frame.shape[:2]
        prm = np.zeros((n, 7), np.float32)
        for i, (x0, y0, x1, y1) in enumerate(rects):
            cw = max(1e-3, float(x1 - x0))
            s = KPT_IN_W / cw
            prm[i] = (x0, y0, x1, y1, s,
                      KPT_IN_W * 0.5 - s * cw * 0.5,
                      KPT_IN_H * 0.5 - s * float(y1 - y0) * 0.5)
        frame = np.ascontiguousarray(frame)
        dev_f = self._alloc("frame", frame.nbytes)
        self._c.cudaMemcpy(dev_f, frame.ctypes.data, frame.nbytes,
                           self._c.cudaMemcpyKind.cudaMemcpyHostToDevice)
        dev_p = self._alloc("prm", prm.nbytes)
        self._c.cudaMemcpy(dev_p, prm.ctypes.data, prm.nbytes,
                           self._c.cudaMemcpyKind.cudaMemcpyHostToDevice)
        dev_out = self._alloc("out", n * 3 * KPT_IN_H * KPT_IN_W * 4)
        grid = int((n * KPT_IN_H * KPT_IN_W + 255) // 256)
        launch(self._core_stream, LaunchConfig(grid=(grid,), block=(256,)), self._ker,
               int(dev_f), int(dev_out), int(dev_p),
               np.int32(n), np.int32(KPT_IN_H), np.int32(KPT_IN_W),
               np.int32(fh), np.int32(fw))
        return dev_out, (n, 3, KPT_IN_H, KPT_IN_W), prm


def make_trt_kpt_gpu(model_dir, device, run_mode, batch_size, args, use_dark=False):
    """KeyPointDetector TRT 版：直接吃整帧 + 裁剪框，预处理与引擎输入全在显存里完成。"""
    from python.keypoint_infer import KeyPointDetector

    class TRTKeyPointDetectorGPU(KeyPointDetector):

        def __init__(self):
            super().__init__(
                model_dir,
                device=device,
                run_mode=run_mode,
                batch_size=batch_size,
                trt_min_shape=args.trt_min_shape,
                trt_max_shape=args.trt_max_shape,
                trt_opt_shape=args.trt_opt_shape,
                trt_calib_mode=args.trt_calib_mode,
                cpu_threads=args.cpu_threads,
                enable_mkldnn=args.enable_mkldnn,
                use_dark=use_dark)
            self.runner = TrtRunner(ENGINE["kpt"])
            self.cropper = GpuKptCropper()

        def predict_frame(self, frame, rects):
            """frame: HxWx3 uint8；rects: [(x0,y0,x1,y1)]。返回与 predict_image 同结构。"""
            n = len(rects)
            if n == 0:
                return dict(keypoint=np.zeros((0, 17, 3)), score=np.zeros((0, 17)))
            dev_img, shape, _ = self.cropper.build(frame, rects)
            item_bytes = 3 * KPT_IN_H * KPT_IN_W * 4
            kpts, scores = [], []
            bs = self.batch_size
            for st in range(0, n, bs):
                m = min(bs, n - st)
                outs = self.runner.run_dev(
                    {"image": (int(dev_img) + st * item_bytes, (m, 3, KPT_IN_H, KPT_IN_W))})
                heat = outs[0][:m]
                for i in range(m):
                    x0, y0, x1, y1 = rects[st + i]
                    im_shape = np.array([[float(y1 - y0), float(x1 - x0)]], np.float32)
                    res = self.postprocess({"im_shape": im_shape},
                                           {"heatmap": heat[i:i + 1], "masks": None})
                    kpts.append(res["keypoint"])
                    scores.append(res["score"])
            return dict(keypoint=np.concatenate(kpts, 0),
                        score=np.concatenate(scores, 0))

    return TRTKeyPointDetectorGPU()
