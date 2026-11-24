#pragma once
#include <opencv2/core.hpp>
#include <vector>

// ByteTrack-style multi-object tracker (single class: pedestrian).
// Two-stage association: high-conf dets first, then low-conf to recover occluded.
struct Trk {
    int id = 0;
    cv::Rect2f tlwh;
    float score = 0.f;
    int age = 0;          // frames since last update
    int hits = 0;         // total matched times
    bool activated = false;
    cv::Point2f velocity; // simple constant-velocity prediction
};

class ByteTracker {
public:
    // defaults aligned to official PP-Human v2.9 tracker_config.yml:
    // BOTSORT track_high_thresh 0.3 / track_low_thresh 0.2 /
    // new_track_thresh 0.4 / track_buffer(maxAge) 30
    ByteTracker(float highThresh = 0.3f, float lowThresh = 0.2f,
                float matchIou = 0.3f, int maxAge = 30, int minHits = 3,
                float newThresh = 0.4f)
        : highThresh_(highThresh), lowThresh_(lowThresh),
          matchIou_(matchIou), maxAge_(maxAge), minHits_(minHits),
          newThresh_(newThresh) {}

    // dets: [x1,y1,x2,y2,score] in image coords. Returns activated tracks.
    std::vector<Trk> update(const std::vector<cv::Rect2f> &dets,
                            const std::vector<float> &scores);

    void reset() { tracks_.clear(); nextId_ = 1; frameId_ = 0; }

private:
    float highThresh_, lowThresh_, matchIou_, newThresh_;
    int maxAge_, minHits_;
    int nextId_ = 1;
    int frameId_ = 0;
    std::vector<Trk> tracks_;

    cv::Rect2f predict(const Trk &t) const;
    static float iou(const cv::Rect2f &a, const cv::Rect2f &b);
    // greedy IoU matching: dets vs arbitrary rects -> pairs (detIdx, rectIdx)
    static std::vector<std::pair<int,int>> match(
        const std::vector<cv::Rect2f> &dets,
        const std::vector<cv::Rect2f> &rects, float thr);
};
