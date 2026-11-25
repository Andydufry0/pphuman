#include "KptModel.h"
#include <opencv2/imgproc.hpp>

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
