#pragma once
#include "TrtEngine.h"
#include <mutex>
#include <vector>

// ---- HRNet keypoint wrapper (top-down, per-person crop 256x192) ----
// Industrial batched inference (official PP-Human batch_size=16): all person
// crops go to the GPU in ONE [n,3,256,192] pass instead of n separate calls.
class KptModel {
public:
    bool load(const std::string &enginePath);
    bool loadOnce(const std::string &enginePath);
    bool ready() const { return eng_.ready(); }
    // crop: person bbox regions from original frame; out aligned with crops
    bool inferBatch(const cv::Mat &bgr, const std::vector<cv::Rect2f> &bboxes,
                    std::vector<std::vector<cv::Point3f>> &out);
    // process-wide shared instance (stateless batched keypoints)
    static KptModel &shared();
private:
    TrtEngine eng_;
    std::mutex mtx_;   // full infer transaction (crop -> heatmap -> decode)
    // reused buffers (high-frequency path: no per-call heap churn)
    cv::Mat crop_;
    std::vector<float> input_, heat_;
};
