# -*- coding: utf-8 -*-
"""Convert PaddleDetection ONNX (tensorrt backend) -> TRT FP16 engines (TRT 10.16)."""
import os, sys
import tensorrt as trt

ONNX_DIR = "models/paddle_onnx"
OUT_DIR  = "models"
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')   # register BatchedNMS etc.

# name -> (onnx file, {input_name: (min_shape, opt_shape, max_shape)})
# kpt: dynamic batch [1..16] — one batched GPU pass for all persons
#      (official PP-Human KPT batch_size=16, DeepStream-style secondary batching)
MODELS = {
    "mot":     ("mot_trt.onnx",     {"image": ([1,3,640,640], [1,3,640,640], [1,3,640,640]),
                                     "scale_factor": ([1,2], [1,2], [1,2])}),
    "kpt":     ("kpt_trt.onnx",     {"image": ([1,3,256,192], [16,3,256,192], [16,3,256,192])}),
    "smoking": ("smoking_trt.onnx", {"image": ([1,3,640,640], [1,3,640,640], [1,3,640,640]),
                                     "scale_factor": ([1,2], [1,2], [1,2])}),
    "ppTSM":   ("ppTSM_trt.onnx",   {"data_batch_0": ([1,8,3,320,320], [1,8,3,320,320], [1,8,3,320,320])}),
    "STGCN":   ("STGCN_trt.onnx",   {"data_batch_0": ([1,2,50,17,1], [1,2,50,17,1], [1,2,50,17,1])}),
}

def build(name, onnx_file, shapes):
    path = os.path.join(ONNX_DIR, onnx_file)
    out  = os.path.join(OUT_DIR, name + ".engine")
    print("="*60, flush=True)
    print("[build]", name, "<-", path, flush=True)
    if not os.path.exists(path):
        print("  skip: not found"); return

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)
    with open(path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  [ERR]", str(parser.get_error(i))[:300])
            print("  FAILED parse"); return

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        nm = inp.name
        if nm in shapes:
            mn, opt, mx = shapes[nm]
            profile.set_shape(nm, mn, opt, mx)
        else:
            print("  WARN: no shape for", nm)
    config.add_optimization_profile(profile)

    print("  building FP16 ...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  FAILED build"); return
    with open(out, "wb") as f:
        f.write(bytearray(serialized))
    print("  saved", out, "(%d bytes)" % serialized.nbytes, flush=True)

only = sys.argv[1] if len(sys.argv) > 1 else None
for name, (onnx_file, shapes) in MODELS.items():
    if only and name != only: continue
    build(name, onnx_file, shapes)
