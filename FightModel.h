#pragma once
#include "ActionModels.h"
#include <atomic>
#include <vector>

// ---- ppTSM video classifier (fight, 8-frame clip) ----
// Matches Python pipeline: every 7th frame sampled -> ShortSizeScale(340,
// stretched to 4:3) -> center crop 320 -> RGB ImageNet norm -> [1,8,3,320,320].
// Logits [1,2] -> softmax; argmax==1 means Fight (Python semantics).
class FightModel {
public:
    FightModel() : se_(SharedActionEngine::fight()) {}
    bool load(const std::string &enginePath);
    bool ready() const { return se_.eng.ready(); }
    // sample every N frames (runtime-adjustable)
    void setSampleFreq(int f) { if (f >= 1) sampleFreq_.store(f); }
    // Feed every frame; internally samples and infers when clip is full.
    // thFight: P(fight) threshold for the positive decision.
    // Returns true when a fresh decision was made this frame.
    bool pushFrame(const cv::Mat &bgr, float thFight, bool &fight, float &score);
private:
    bool infer(float &score);   // softmax P(fight), clears clip
    SharedActionEngine &se_;    // process-wide engine + transaction lock
    std::vector<cv::Mat> clip_; // sampled frames (short-size scaled)
    std::vector<float> input_;  // reused TCHW buffer
    long long cnt_ = 0;
    std::atomic<int> sampleFreq_{7};
};
