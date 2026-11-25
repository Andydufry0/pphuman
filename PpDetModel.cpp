#include "PpDetModel.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

// ==================== PpDetModel ====================

PpDetModel &PpDetModel::sharedMot()   { static PpDetModel m; return m; }
PpDetModel &PpDetModel::sharedSmoke() { static PpDetModel m; return m; }

bool PpDetModel::load(const std::string &enginePath, bool imagenetNorm) {
    std::lock_guard<std::mutex> lk(mtx_);
    return loadUnlocked(enginePath, imagenetNorm);
}

bool PpDetModel::loadOnce(const std::string &enginePath, bool imagenetNorm) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (eng_.ready()) return true;
    return loadUnlocked(enginePath, imagenetNorm);
}

bool PpDetModel::loadUnlocked(const std::string &enginePath, bool imagenetNorm) {
    norm_ = imagenetNorm;
    if (!eng_.load(enginePath)) return false;
    nvinfer1::Dims d = eng_.inShape("image");
    if (d.nbDims == 4) { size_ = (int)d.d[3]; }
    std::cout << "[PpDet] " << enginePath << " input " << size_ << "x" << size_
              << " norm=" << norm_ << "\n";
    return true;
}

static void chwNormalize(const cv::Mat &im, std::vector<float> &out,
                         const float mean[3], const float stdv[3]) {
    // im: NxN BGR uint8 -> CHW RGB normalized
    const int N = im.rows, plane = N * N;
    out.resize(3 * plane);
    for (int c = 0; c < 3; ++c) {
        float *dst = out.data() + (2 - c) * plane;   // BGR src -> RGB plane
        for (int y = 0; y < N; ++y) {
            const cv::Vec3b *row = im.ptr<cv::Vec3b>(y);
            for (int x = 0; x < N; ++x) {
                float v = row[x][c] * (1.0f / 255.0f);
                dst[y * N + x] = (v - mean[2 - c]) / stdv[2 - c];
            }
        }
    }
}

std::vector<DetResult> PpDetModel::infer(const cv::Mat &bgr, float scoreThresh) {
    std::vector<DetResult> results;
    if (!eng_.ready() || bgr.empty()) return results;
    std::lock_guard<std::mutex> lk(mtx_);   // full transaction, shared engine

    const int N = size_;
    cv::resize(bgr, im_, cv::Size(N, N), 0, 0, cv::INTER_CUBIC);

    if (norm_) {
        static const float mean[3] = {0.485f, 0.456f, 0.406f};
        static const float stdv[3] = {0.229f, 0.224f, 0.225f};
        chwNormalize(im_, input_, mean, stdv);
    } else {
        // raw BGR 0-255, HWC -> CHW
        const int plane = N * N;
        input_.resize(3 * plane);
        for (int c = 0; c < 3; ++c) {
            float *dst = input_.data() + c * plane;
            for (int y = 0; y < N; ++y) {
                const cv::Vec3b *row = im_.ptr<cv::Vec3b>(y);
                for (int x = 0; x < N; ++x)
                    dst[y * N + x] = (float)row[x][c];
            }
        }
    }

    float sf[2] = { (float)N / bgr.rows, (float)N / bgr.cols };
    eng_.setInput("image", input_.data(), input_.size());
    eng_.setInput("scale_factor", sf, 2);
    if (!eng_.infer()) return results;

    eng_.getOutput("multiclass_nms3_0.tmp_0", boxes_);
    const int M = (int)boxes_.size() / 6;
    for (int i = 0; i < M; ++i) {
        const float *p = boxes_.data() + i * 6;
        float score = p[1];
        if (score < scoreThresh) continue;
        DetResult r;
        r.cls = (int)p[0];
        r.score = score;
        r.box = cv::Rect2f(p[2], p[3], p[4] - p[2], p[5] - p[3]);
        results.push_back(r);
    }
    return results;
}
