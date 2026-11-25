#pragma once
#include "ModelTypes.h"
#include "TrtEngine.h"
#include <mutex>
#include <string>
#include <vector>

// ---- PaddleDetection PP-YOLOE detector wrapper (MOT / smoking) ----
// Preprocess (verified on engine): direct resize to NxN, BGR 0-255 raw
// (smoking variant: ImageNet-normalized RGB), scale_factor=[N/h, N/w].
// Output boxes are already in original frame coords: [cls, score, x1,y1,x2,y2].
class PpDetModel {
public:
    bool load(const std::string &enginePath, bool imagenetNorm);
    // idempotent multi-channel load: shared instance loads exactly once
    bool loadOnce(const std::string &enginePath, bool imagenetNorm);
    bool ready() const { return eng_.ready(); }
    std::vector<DetResult> infer(const cv::Mat &bgr, float scoreThresh);
    int inputSize() const { return size_; }

    // process-wide shared instances (stateless models: MOT / smoking).
    // 16 channels sharing one engine = 1.5 GB VRAM instead of 24 GB;
    // inference is serialized by the model mutex + global GPU lock
    static PpDetModel &sharedMot();
    static PpDetModel &sharedSmoke();

private:
    TrtEngine eng_;
    bool norm_ = false;
    int size_ = 640;
    std::mutex mtx_;   // full infer transaction (preprocess -> nms)
    bool loadUnlocked(const std::string &enginePath, bool imagenetNorm);
    // reused buffers (per-frame path: no heap churn)
    cv::Mat im_;
    std::vector<float> input_, boxes_;
};
