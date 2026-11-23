#include "ByteTracker.h"
#include <algorithm>
#include <cmath>
#include <limits>

float ByteTracker::iou(const cv::Rect2f &a, const cv::Rect2f &b) {
    float ix1 = std::max(a.x, b.x), iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.x + a.width, b.x + b.width);
    float iy2 = std::min(a.y + a.height, b.y + b.height);
    float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 1e-6f ? inter / uni : 0.f;
}

cv::Rect2f ByteTracker::predict(const Trk &t) const {
    // constant velocity prediction
    return cv::Rect2f(t.tlwh.x + t.velocity.x, t.tlwh.y + t.velocity.y,
                      t.tlwh.width, t.tlwh.height);
}

std::vector<std::pair<int,int>> ByteTracker::match(
    const std::vector<cv::Rect2f> &dets,
    const std::vector<cv::Rect2f> &rects, float thr)
{
    // build IoU matrix, greedy highest-first
    std::vector<std::tuple<float,int,int>> cand;
    for (size_t d = 0; d < dets.size(); ++d) {
        for (size_t t = 0; t < rects.size(); ++t) {
            float v = iou(dets[d], rects[t]);
            if (v >= thr) cand.emplace_back(v, (int)d, (int)t);
        }
    }
    std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
        return std::get<0>(a) > std::get<0>(b);
    });
    std::vector<bool> dUsed(dets.size(), false), tUsed(rects.size(), false);
    std::vector<std::pair<int,int>> pairs;
    for (const auto &c : cand) {
        int d = std::get<1>(c), t = std::get<2>(c);
        if (dUsed[d] || tUsed[t]) continue;
        dUsed[d] = tUsed[t] = true;
        pairs.emplace_back(d, t);
    }
    return pairs;
}

std::vector<Trk> ByteTracker::update(
    const std::vector<cv::Rect2f> &dets, const std::vector<float> &scores)
{
    frameId_++;

    // split high / low confidence
    std::vector<int> high, low;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (scores[i] >= highThresh_) high.push_back((int)i);
        else if (scores[i] >= lowThresh_) low.push_back((int)i);
    }
    std::vector<cv::Rect2f> highDets, lowDets;
    for (int i : high) highDets.push_back(dets[i]);
    for (int i : low)  lowDets.push_back(dets[i]);

    // predicted positions for existing tracks
    std::vector<cv::Rect2f> predicted(tracks_.size());
    for (size_t t = 0; t < tracks_.size(); ++t) predicted[t] = predict(tracks_[t]);

    std::vector<bool> trkMatched(tracks_.size(), false);
    std::vector<bool> highMatched(highDets.size(), false);

    // stage 1: high-conf dets vs tracks
    auto p1 = match(highDets, predicted, matchIou_);
    for (auto &pr : p1) {
        int d = pr.first, t = pr.second;
        const auto &det = highDets[d];
        const auto &pred = predicted[t];
        Trk &tk = tracks_[t];
        // update velocity (smoothed)
        tk.velocity.x = 0.5f * tk.velocity.x + 0.5f * (det.x - pred.x);
        tk.velocity.y = 0.5f * tk.velocity.y + 0.5f * (det.y - pred.y);
        tk.tlwh = det;
        tk.score = scores[high[d]];
        tk.age = 0;
        tk.hits++;
        tk.activated = true;
        trkMatched[t] = true;
        highMatched[d] = true;
    }

    // stage 2: low-conf dets vs unmatched tracks (recover occluded)
    std::vector<int> unmatchedTracks;
    for (size_t t = 0; t < tracks_.size(); ++t)
        if (!trkMatched[t]) unmatchedTracks.push_back((int)t);
    std::vector<cv::Rect2f> unTrkRects;
    for (int t : unmatchedTracks) unTrkRects.push_back(predicted[t]);
    auto p2 = match(lowDets, unTrkRects, matchIou_);
    for (auto &pr : p2) {
        int d = pr.first, t = unmatchedTracks[pr.second];
        const auto &det = lowDets[d];
        Trk &tk = tracks_[t];
        tk.tlwh = det;
        tk.age = 0;
        tk.hits++;
        trkMatched[t] = true;
    }

    // create new tracks from unmatched dets (official new_track_thresh 0.4)
    for (size_t d = 0; d < highDets.size(); ++d) {
        if (highMatched[d]) continue;
        if (scores[high[d]] < newThresh_) continue;   // BOTSORT new_track_thresh
        Trk tk;
        tk.id = nextId_++;
        tk.tlwh = highDets[d];
        tk.score = scores[high[d]];
        tk.age = 0;
        tk.hits = 1;
        tk.activated = (tk.hits >= minHits_);
        tracks_.push_back(tk);
        trkMatched.push_back(true);   // newly created track counts as matched
    }

    // age unmatched tracks, remove dead ones
    std::vector<Trk> kept;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        Trk &tk = tracks_[i];
        if (trkMatched[i]) {
            kept.push_back(tk);
        } else {
            tk.age++;
            tk.velocity.x *= 0.9f;
            tk.velocity.y *= 0.9f;
            if (tk.age <= maxAge_) kept.push_back(tk);
        }
    }
    tracks_ = kept;

    // return activated tracks only
    std::vector<Trk> out;
    for (const auto &tk : tracks_)
        if (tk.activated && tk.age == 0) out.push_back(tk);
    return out;
}
