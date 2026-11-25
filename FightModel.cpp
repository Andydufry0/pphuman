#include "FightModel.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

// ==================== FightModel (ppTSM) ====================

bool FightModel::load(const std::string &enginePath) {
    clip_.reserve(8);
    return se_.loadOnce(enginePath);   // engine loaded once, shared across switches
}

bool FightModel::pushFrame(const cv::Mat &bgr, float thFight, bool &fight, float &score) {
    fight = false; score = 0.f;
    if (!se_.eng.ready() || bgr.empty()) return false;
    if (++cnt_ % sampleFreq_.load() != 0) return false;

    // ShortSizeScale(340, fixed_ratio 4:3): w<=h -> 340x453, w>h -> 453x340
    int sw, sh;
    if (bgr.cols <= bgr.rows) { sw = 340; sh = 453; } else { sw = 453; sh = 340; }
    cv::Mat scaled;
    cv::resize(bgr, scaled, cv::Size(sw, sh), 0, 0, cv::INTER_LINEAR);
    clip_.push_back(scaled);
    if ((int)clip_.size() < 8) return false;

    bool ok = infer(score);
    clip_.clear();                               // next clip
    if (!ok) return false;
    fight = (score >= thFight);                  // positive if P(fight) >= threshold
    return true;
}

bool FightModel::infer(float &score) {
    // [1,8,3,320,320] TCHW, RGB ImageNet normalization
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    const int S = 320;
    input_.resize((size_t)8 * 3 * S * S);
    for (int t = 0; t < 8; ++t) {
        const cv::Mat &m = clip_[t];
        cv::Rect crop((m.cols - S) / 2, (m.rows - S) / 2, S, S);  // center crop
        cv::Mat cc = m(crop);
        float *base = input_.data() + (size_t)t * 3 * S * S;
        for (int c = 0; c < 3; ++c) {            // c: 0=R,1=G,2=B (channel of RGB)
            float *dst = base + c * S * S;
            for (int y = 0; y < S; ++y) {
                const cv::Vec3b *row = cc.ptr<cv::Vec3b>(y);
                for (int x = 0; x < S; ++x) {
                    float v = row[x][2 - c] * (1.0f / 255.0f);  // BGR -> RGB
                    dst[y * S + x] = (v - mean[c]) / stdv[c];
                }
            }
        }
    }
    // full transaction under shared lock: device buffers inside eng_ must not
    // be overwritten by a concurrent channel between setInput and getOutput
    {
        std::lock_guard<std::mutex> lk(se_.mtx);
        se_.eng.setInput("data_batch_0", input_.data(), input_.size());
        if (!se_.eng.infer()) return false;
        auto out = se_.eng.getOutput("linear_2.tmp_1");
        if (out.size() < 2) return false;
        float e0 = std::exp(out[0]), e1 = std::exp(out[1]);
        score = e1 / (e0 + e1);                  // softmax P(fight)
    }
    return true;
}
