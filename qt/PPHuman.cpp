#include "PPHuman.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <unordered_set>

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
                dst[y * N + x] = (v - mean[c]) / stdv[c];
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

// ==================== KptModel (batched) ====================

KptModel &KptModel::shared() { static KptModel m; return m; }

bool KptModel::load(const std::string &enginePath) {
    std::lock_guard<std::mutex> lk(mtx_);
    return eng_.load(enginePath);
}

bool KptModel::loadOnce(const std::string &enginePath) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (eng_.ready()) return true;
    return eng_.load(enginePath);
}

bool KptModel::inferBatch(const cv::Mat &bgr, const std::vector<cv::Rect2f> &bboxes,
                          std::vector<std::vector<cv::Point3f>> &out) {
    out.assign(bboxes.size(), {});
    if (!eng_.ready() || bgr.empty()) return false;
    std::lock_guard<std::mutex> lk(mtx_);   // full transaction, shared engine

    const int W = 192, H = 256, plane = H * W;
    const int n = (int)bboxes.size();
    if (n <= 0) return false;

    // build [n,3,256,192] CHW batch in one reused buffer
    input_.resize((size_t)n * 3 * plane);
    for (int i = 0; i < n; ++i) {
        cv::Rect2f rc = bboxes[i] & cv::Rect2f(0, 0, (float)bgr.cols, (float)bgr.rows);
        if (rc.width < 2 || rc.height < 2) continue;
        cv::Point2f src[3] = {
            {rc.x, rc.y},
            {rc.x + rc.width, rc.y},
            {rc.x, rc.y + rc.height}};
        cv::Point2f dst[3] = {
            {0, 0},
            {(float)W, 0},
            {0, (float)H}};
        cv::Mat M = cv::getAffineTransform(src, dst);
        cv::warpAffine(bgr, crop_, M, cv::Size(W, H), cv::INTER_LINEAR);
        float *base = input_.data() + (size_t)i * 3 * plane;
        for (int c = 0; c < 3; ++c) {
            float *dst2 = base + c * plane;
            for (int y = 0; y < H; ++y) {
                const cv::Vec3b *row = crop_.ptr<cv::Vec3b>(y);
                for (int x = 0; x < W; ++x)
                    dst2[y * W + x] = (float)row[x][c];
            }
        }
    }

    nvinfer1::Dims d;
    d.nbDims = 4;
    d.d[0] = n; d.d[1] = 3; d.d[2] = H; d.d[3] = W;
    if (!eng_.setInputShape("image", d)) return false;
    eng_.setInput("image", input_.data(), input_.size());
    if (!eng_.infer()) return false;

    eng_.getOutput("conv2d_585.tmp_1", heat_);   // [n,17,64,48]
    if (heat_.empty()) return false;

    const int hmH = 64, hmW = 48;
    for (int i = 0; i < n; ++i) {
        cv::Rect2f rc = bboxes[i] & cv::Rect2f(0, 0, (float)bgr.cols, (float)bgr.rows);
        if (rc.width < 2 || rc.height < 2) continue;
        // heatmap coords -> crop coords -> original frame (inverse affine)
        cv::Point2f src[3] = {
            {rc.x, rc.y},
            {rc.x + rc.width, rc.y},
            {rc.x, rc.y + rc.height}};
        cv::Point2f dst[3] = {
            {0, 0},
            {(float)W, 0},
            {0, (float)H}};
        cv::Mat Minv = cv::getAffineTransform(dst, src);
        const float *hbuf = heat_.data() + (size_t)i * 17 * hmH * hmW;
        auto &kp = out[i];
        kp.resize(17);
        for (int k = 0; k < 17; ++k) {
            const float *ch = hbuf + (size_t)k * hmH * hmW;
            int best = 0;
            for (int j = 1; j < hmH * hmW; ++j) if (ch[j] > ch[best]) best = j;
            int hy = best / hmW, hx = best % hmW;
            float conf = ch[best];
            // sub-pixel refine: weighted mean of 5x5 neighborhood
            float sx = 0, sy = 0, sw = 0;
            for (int dy = -2; dy <= 2; ++dy) {
                int yy = hy + dy; if (yy < 0 || yy >= hmH) continue;
                for (int dx = -2; dx <= 2; ++dx) {
                    int xx = hx + dx; if (xx < 0 || xx >= hmW) continue;
                    float v = ch[yy * hmW + xx];
                    sx += v * xx; sy += v * yy; sw += v;
                }
            }
            float cx_ = sw > 1e-6f ? sx / sw : (float)hx;
            float cy_ = sw > 1e-6f ? sy / sw : (float)hy;
            float px = cx_ / hmW * W;
            float py = cy_ / hmH * H;
            double ox = Minv.at<double>(0,0)*px + Minv.at<double>(0,1)*py + Minv.at<double>(0,2);
            double oy = Minv.at<double>(1,0)*px + Minv.at<double>(1,1)*py + Minv.at<double>(1,2);
            kp[k] = cv::Point3f((float)ox, (float)oy, conf);
        }
    }
    return true;
}

// ==================== Shared action engines (fight / fall) ====================

SharedActionEngine &SharedActionEngine::fight() {
    static SharedActionEngine e; return e;
}
SharedActionEngine &SharedActionEngine::fall() {
    static SharedActionEngine e; return e;
}
bool SharedActionEngine::loadOnce(const std::string &path) {
    std::lock_guard<std::mutex> lk(mtx);
    if (eng.ready()) return true;   // already loaded by another channel/switch
    bool ok = eng.load(path);
    if (ok) std::cout << "[SharedAction] loaded " << path << "\n";
    return ok;
}

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
        if ((int)s.frames.size() >= minFrames_)
            out.push_back(inferOne(s, ids[i]));
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
