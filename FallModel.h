#pragma once
#include "ActionModels.h"
#include <deque>
#include <unordered_map>
#include <vector>

// ---- STGCN skeleton classifier (falling, rolling 50-frame keypoint window) ----
// Coords normalized per-frame to person box scaled to 384x512 (official
// parse_mot_keypoint). Input [1,2,50,17,1]; logits [1,2]; argmax==0 means Falling.
class FallModel {
public:
    FallModel() : se_(SharedActionEngine::fall()) {}
    struct Res { int id; bool falling; float score; };  // score = softmax P(fall)
    bool load(const std::string &enginePath);
    bool ready() const { return se_.eng.ready(); }
    // minimum real frames before first inference (short windows are zero-padded)
    void setMinFrames(int f) { if (f >= 10 && f <= 50) minFrames_ = f; }
    // Call on every KPT frame (ids aligned with boxes and kpts).
    // Returns one result per track with enough window history.
    std::vector<Res> update(const std::vector<int> &ids,
                            const std::vector<cv::Rect2f> &boxes,
                            const std::vector<std::vector<cv::Point3f>> &kpts);
private:
    struct FrameSample { float kpt[34]; float box[4]; };  // 17*(x,y), x1y1x2y2
    struct Seq { std::deque<FrameSample> frames; };       // rolling, max 50
    SharedActionEngine &se_;    // process-wide engine + transaction lock
    std::unordered_map<int, Seq> seqs_;
    std::vector<float> input_;   // reused [1,2,50,17,1] buffer
    int minFrames_ = 50;   // full 50-frame window (official); short windows zero-pad -> false falls
    Res inferOne(const Seq &s, int id);
};
