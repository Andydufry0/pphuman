#include "FallModel.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

// ==================== FallModel (STGCN) ====================

bool FallModel::load(const std::string &enginePath) {
    return se_.loadOnce(enginePath);   // engine loaded once, shared across switches
}

std::vector<FallModel::Res> FallModel::update(const std::vector<int> &ids,
        const std::vector<cv::Rect2f> &boxes,
        const std::vector<std::vector<cv::Point3f>> &kpts) {
    std::vector<Res> out;
    if (!se_.eng.ready()) return out;

    std::unordered_set<int> updated;
    for (size_t i = 0; i < ids.size() && i < boxes.size() && i < kpts.size(); ++i) {
        if (kpts[i].size() < 17) continue;
        updated.insert(ids[i]);
        Seq &s = seqs_[ids[i]];
        FrameSample fs;
        for (int k = 0; k < 17; ++k) {
            fs.kpt[k * 2]     = kpts[i][k].x;
            fs.kpt[k * 2 + 1] = kpts[i][k].y;
        }
        fs.box[0] = boxes[i].x;
        fs.box[1] = boxes[i].y;
        fs.box[2] = boxes[i].x + boxes[i].width;
        fs.box[3] = boxes[i].y + boxes[i].height;
        s.frames.push_back(fs);
        if (s.frames.size() > 50) s.frames.pop_front();   // rolling window
        if ((int)s.frames.size() >= minFrames_) {
            out.push_back(inferOne(s, ids[i]));
            // official KeyPointBuff: a full window is popped once and then the
            // buffer is reset -> one decision per 50 frames (≈2 s @25 fps),
            // NOT a sliding window that infers on every frame. Sliding made
            // the same ambiguous posture trigger 25 decisions per second, so
            // keypoint jitter alone could satisfy any debounce counter.
            if (minFrames_ >= 50) s.frames.clear();
        }
    }
    // interrupted trackers: window already inferred every frame, just drop
    for (auto it = seqs_.begin(); it != seqs_.end();) {
        if (updated.count(it->first)) { ++it; continue; }
        it = seqs_.erase(it);
    }
    return out;
}

FallModel::Res FallModel::inferOne(const Seq &s, int id) {
    Res r; r.id = id; r.falling = false; r.score = 0.f;
    static const float CW = 384.f, CH = 512.f;   // coord_size
    input_.assign((size_t)2 * 50 * 17, 0.f);     // AutoPadding: data front, zeros tail
    float *xch = input_.data();
    float *ych = input_.data() + 50 * 17;
    int t = 0;
    for (const FrameSample &fs : s.frames) {
        float tlx = fs.box[0], tly = fs.box[1];
        float w = std::max(fs.box[2] - tlx, 1e-3f), h = std::max(fs.box[3] - tly, 1e-3f);
        for (int k = 0; k < 17; ++k) {
            xch[t * 17 + k] = (fs.kpt[k * 2]     - tlx) / w * CW;
            ych[t * 17 + k] = (fs.kpt[k * 2 + 1] - tly) / h * CH;
        }
        ++t;
    }
    // full transaction under shared lock (see FightModel::infer)
    {
        std::lock_guard<std::mutex> lk(se_.mtx);
        se_.eng.setInput("data_batch_0", input_.data(), input_.size());
        if (!se_.eng.infer()) return r;
        auto logits = se_.eng.getOutput("reshape2_34.tmp_0");
        if (logits.size() < 2) return r;
        float e0 = std::exp(logits[0]), e1 = std::exp(logits[1]);
        r.falling = (logits[0] > logits[1]);     // class 0 = Falling
        r.score = e0 / (e0 + e1);
    }
    return r;
}
