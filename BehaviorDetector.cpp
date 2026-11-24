#include "BehaviorDetector.h"
#include <cmath>
#include <algorithm>

bool BehaviorDetector::inRegion(float cx, float cy, int imgW, int imgH) const {
    if (!hasRegion_) return true;
    return cx >= rx1_ && cx <= rx2_ && cy >= ry1_ && cy <= ry2_;
}

void BehaviorDetector::reset() {
    loiters_.clear();
    runs_.clear();
    postures_.clear();
    lastAlert_.clear();
    crowdStart_ = -1;
    crowdActive_ = crowdWasActive_ = false;
    pendingKpts_.clear();
    kptBoxes_.clear();
}

bool BehaviorDetector::cooldownOk(int type, int trackId, double now) {
    int key = type * 100000 + trackId;
    auto it = lastAlert_.find(key);
    if (it != lastAlert_.end() && now - it->second < alertCooldown_)
        return false;
    lastAlert_[key] = now;
    // periodic cleanup of stale keys
    if (lastAlert_.size() > 256) {
        for (auto it2 = lastAlert_.begin(); it2 != lastAlert_.end();) {
            if (now - it2->second > 2.0 * alertCooldown_)
                it2 = lastAlert_.erase(it2);
            else ++it2;
        }
    }
    return true;
}

std::vector<BehaviorEvent> BehaviorDetector::update(
    const std::vector<TrackBox> &boxes, int frameId, int imgW, int imgH)
{
    std::vector<BehaviorEvent> events;
    double now = frameId / fps_;

    // ---------- 1. Crowd gathering ----------
    int personCount = 0;
    for (const auto &b : boxes) {
        if (b.score >= minScoreCrowd_ && inRegion(b.cx(), b.cy(), imgW, imgH))
            personCount++;
    }
    bool isCrowd = personCount >= crowdThreshold_;
    if (isCrowd) {
        if (crowdStart_ < 0) crowdStart_ = now;
        else if (now - crowdStart_ >= crowdAlertSec_) crowdActive_ = true;
    } else {
        crowdStart_ = -1;
        crowdActive_ = false;
    }
    if (crowdActive_ || crowdWasActive_ || isCrowd) {
        BehaviorEvent e;
        e.type = BehaviorEvent::CROWD;
        e.trackId = -1;
        e.active = crowdActive_;
        e.alert = crowdActive_ && !crowdWasActive_ &&
                  cooldownOk(BehaviorEvent::CROWD, -1, now);
        e.value = (float)personCount;
        e.detail = "count=" + std::to_string(personCount);
        events.push_back(e);
    }
    crowdWasActive_ = crowdActive_;

    // ---------- track id sets for cleanup ----------
    std::vector<int> currentIds;
    currentIds.reserve(boxes.size());

    for (const auto &b : boxes) {
        int id = b.id;
        currentIds.push_back(id);
        float cx = b.cx(), cy = b.cy();

        // ---------- 2. Loitering ----------
        if (b.score >= minScoreLoiter_ && inRegion(cx, cy, imgW, imgH)) {
            auto &ls = loiters_[id];
            if (!ls.hasPos) {
                ls.firstSeen = now;
                ls.firstPos = cv::Point2f(cx, cy);
                ls.lastPos = cv::Point2f(cx, cy);
                ls.hasPos = true;
            }
            ls.lastPos = cv::Point2f(cx, cy);
            double netDisp = std::hypot(cx - ls.firstPos.x, cy - ls.firstPos.y);
            double duration = now - ls.firstSeen;
            bool loiter = (duration >= loiterSec_ && netDisp < loiterMovePx_);
            if (loiter || ls.wasLoitering) {
                BehaviorEvent e;
                e.type = BehaviorEvent::LOITER;
                e.trackId = id;
                e.active = loiter;
                e.alert = loiter && !ls.wasLoitering &&
                          cooldownOk(BehaviorEvent::LOITER, id, now);
                e.value = (float)duration;
                e.detail = "id=" + std::to_string(id) + " t=" + std::to_string((int)duration) + "s";
                events.push_back(e);
            }
            ls.wasLoitering = loiter;
        }

        // ---------- 3. Running ----------
        if (b.score >= minScoreRun_) {
            auto &rs = runs_[id];
            rs.positions.push_back(cv::Point2f(cx, cy));
            rs.times.push_back(now);
            if (rs.positions.size() > 10) { rs.positions.pop_front(); rs.times.pop_front(); }
            if (rs.positions.size() >= 2) {
                // net displacement, NOT accumulated path length: summing
                // per-hop distances lets bbox jitter of a stationary person
                // add up to "running". Net (first->last) is robust to jitter.
                double dist = std::hypot(rs.positions.back().x - rs.positions.front().x,
                                         rs.positions.back().y - rs.positions.front().y);
                double dt = rs.times.back() - rs.times.front();
                if (dt > 0 && (int)rs.positions.size() >= runMinFrames_) {
                    double speed = dist / dt;
                    // resolution-normalized threshold: px/s is meaningless across
                    // camera resolutions (400 px/s is a stroll on a 4K frame but
                    // a sprint on 720p). Scale to the 1080p reference height.
                    double speedThr = runSpeedPx_ * (imgH > 0 ? imgH / 1080.0 : 1.0);
                    bool running = speed >= speedThr;
                    if (running || rs.wasRunning) {
                        BehaviorEvent e;
                        e.type = BehaviorEvent::RUN;
                        e.trackId = id;
                        e.active = running;
                        e.alert = running && !rs.wasRunning &&
                                  cooldownOk(BehaviorEvent::RUN, id, now);
                        e.value = (float)speed;
                        e.detail = "id=" + std::to_string(id) + " v=" + std::to_string((int)speed);
                        events.push_back(e);
                    }
                    rs.wasRunning = running;
                }
            }
        }
    }

    // ---------- 4. Abnormal posture (from keypoints) ----------
    if (!pendingKpts_.empty()) {
        size_t n = std::min(pendingKpts_.size(), kptBoxes_.size());
        for (size_t i = 0; i < n; ++i) {
            if (kptBoxes_[i].score < minScorePosture_) continue;
            int id = kptBoxes_[i].id;
            auto pr = analyzePosture(pendingKpts_[i]);
            auto &ps = postures_[id];
            if (pr.abnormal) ++ps.streak; else ps.streak = 0;
            if (pr.abnormal || ps.wasAbnormal) {
                BehaviorEvent e;
                e.type = BehaviorEvent::POSTURE;
                e.trackId = id;
                e.active = pr.abnormal;
                // industrial debouncing: a single tilted frame is not an
                // abnormal posture (fighting/jumping tilt the torso). Require
                // 4 consecutive abnormal KPT frames (~0.4 s @10 Hz) + cooldown.
                e.alert = pr.abnormal && ps.streak >= 4 && !ps.wasAbnormal &&
                          cooldownOk(BehaviorEvent::POSTURE, id, now);
                e.value = pr.angle;
                e.detail = "id=" + std::to_string(id) + " " + pr.type;
                events.push_back(e);
            }
            ps.wasAbnormal = pr.abnormal;
        }
    }
    pendingKpts_.clear();
    kptBoxes_.clear();

    // ---------- cleanup disappeared tracks ----------
    for (auto it = loiters_.begin(); it != loiters_.end(); ) {
        if (std::find(currentIds.begin(), currentIds.end(), it->first) == currentIds.end())
            it = loiters_.erase(it); else ++it;
    }
    for (auto it = runs_.begin(); it != runs_.end(); ) {
        if (std::find(currentIds.begin(), currentIds.end(), it->first) == currentIds.end())
            it = runs_.erase(it); else ++it;
    }
    return events;
}

void BehaviorDetector::updateKeypoints(
    const std::vector<std::vector<cv::Point3f>> &kpts,
    const std::vector<TrackBox> &boxes)
{
    pendingKpts_ = kpts;
    kptBoxes_ = boxes;
}

BehaviorDetector::PostureResult BehaviorDetector::analyzePosture(
    const std::vector<cv::Point3f> &k)
{
    PostureResult r;
    if (k.size() < 17) return r;
    // COCO index: 0 nose, 5 L-shoulder, 6 R-shoulder, 11 L-hip, 12 R-hip
    if (k[0].z < 0.3f || k[5].z < 0.3f || k[6].z < 0.3f) return r;

    cv::Point2f ls(k[5].x, k[5].y), rs(k[6].x, k[6].y);
    cv::Point2f lh(k[11].x, k[11].y), rh(k[12].x, k[12].y);
    cv::Point2f sc((ls.x+rs.x)*0.5f, (ls.y+rs.y)*0.5f);
    cv::Point2f hc((lh.x+rh.x)*0.5f, (lh.y+rh.y)*0.5f);

    cv::Point2f torso(sc.x-hc.x, sc.y-hc.y);   // shoulder - hip (points up when standing)
    cv::Point2f up(0.f, -1.f);
    double tnorm = std::hypot(torso.x, torso.y) + 1e-6;
    double cosA = (torso.x*up.x + torso.y*up.y) / tnorm;
    cosA = std::max(-1.0, std::min(1.0, cosA));
    r.angle = (float)(std::acos(cosA) * 180.0 / CV_PI);
    bool fallen = r.angle > fallAngleDeg_;

    double dl = std::hypot(ls.x-lh.x, ls.y-lh.y);
    double dr = std::hypot(rs.x-rh.x, rs.y-rh.y);
    bool twisted = false;
    if (dl > 0 && dr > 0) {
        double sym = std::abs(dl-dr) / std::max(dl, dr);
        twisted = sym > symmetryThresh_;
    }

    r.abnormal = fallen || twisted;
    if (fallen) r.type = "fallen";
    else if (twisted) r.type = "twisted";
    else r.type = "normal";
    return r;
}
