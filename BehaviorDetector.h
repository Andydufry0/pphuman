#pragma once
#include <opencv2/core.hpp>
#include <unordered_map>
#include <deque>
#include <utility>
#include <string>
#include <vector>

// MOT box: [track_id, class, score, xmin, ymin, w, h]  (matches Python pipeline)
struct TrackBox {
    int id = 0;
    int cls = 0;
    float score = 0.f;
    float x = 0.f, y = 0.f, w = 0.f, h = 0.f;
    float cx() const { return x + w * 0.5f; }
    float cy() const { return y + h * 0.5f; }
};

struct BehaviorEvent {
    enum Type { NONE, CROWD, LOITER, RUN, POSTURE };
    Type type = NONE;
    int trackId = -1;
    bool alert = false;      // rising edge
    bool active = false;     // current state
    std::string detail;
    float value = 0.f;       // count/duration/speed/angle
};

// 4 rule-based behavior detectors ported 1:1 from extended_actions.py
class BehaviorDetector {
public:
    void setFps(double fps) { fps_ = fps > 1e-6 ? fps : 1e-6; }
    // full state reset (stop -> start must not carry stale trackers)
    void reset();

    // Call each frame with tracked boxes. Returns events raised this frame.
    std::vector<BehaviorEvent> update(const std::vector<TrackBox> &boxes,
                                      int frameId, int imgW, int imgH);

    // Abnormal posture needs keypoints (COCO 17), aligned with boxes order.
    void updateKeypoints(const std::vector<std::vector<cv::Point3f>> &kpts,
                         const std::vector<TrackBox> &boxes);

    // Config
    void setRegion(float x1, float y1, float x2, float y2) {
        hasRegion_ = true; rx1_=x1; ry1_=y1; rx2_=x2; ry2_=y2;
    }
    int   crowdThreshold_ = 5;
    float crowdAlertSec_  = 3.0f;
    float loiterSec_      = 10.0f;
    float loiterMovePx_   = 50.0f;
    float runSpeedPx_     = 400.0f;
    int   runMinFrames_   = 5;
    float fallAngleDeg_   = 60.0f;
    float symmetryThresh_ = 0.3f;

    // per-behavior minimum tracked-person score (UI adjustable)
    float minScoreCrowd_   = 0.30f;
    float minScoreLoiter_  = 0.30f;
    float minScoreRun_     = 0.30f;
    float minScorePosture_ = 0.30f;

private:
    struct LoiterState {
        double firstSeen = -1;
        cv::Point2f firstPos, lastPos;
        bool wasLoitering = false;
        bool hasPos = false;
    };
    struct RunState {
        std::deque<cv::Point2f> positions;
        std::deque<double> times;
        bool wasRunning = false;
    };
    struct PostureState { bool wasAbnormal = false; int streak = 0; };

    // crowd
    double crowdStart_ = -1;
    bool crowdActive_ = false, crowdWasActive_ = false;

    std::unordered_map<int, LoiterState> loiters_;
    std::unordered_map<int, RunState> runs_;
    std::unordered_map<int, PostureState> postures_;

    // pending keypoints for posture (set by updateKeypoints, used in update)
    std::vector<std::vector<cv::Point3f>> pendingKpts_;
    std::vector<TrackBox> kptBoxes_;

    double fps_ = 25.0;

    // industrial anti-repeat: same event key (type+track) re-alerts only
    // after the cooldown (NVR "alert interval" standard, 60 s) — threshold
    // hysteresis alone would re-fire on every borderline flip
    double alertCooldown_ = 60.0;
    std::unordered_map<int, double> lastAlert_;
    bool cooldownOk(int type, int trackId, double now);

    bool hasRegion_ = false;
    float rx1_=0, ry1_=0, rx2_=0, ry2_=0;

    // posture analysis for one person; returns abnormal + type + angle
    struct PostureResult { bool abnormal=false; bool fallen=false; std::string type; float angle=0.f; };
    PostureResult analyzePosture(const std::vector<cv::Point3f> &kpts);

    bool inRegion(float cx, float cy, int imgW, int imgH) const;
};
