#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include "TrtEngine.h"

// ---- data types shared across pipeline ----
struct DetResult {
    int cls = 0;
    float score = 0.f;
    cv::Rect2f box;   // original frame coords
};

struct KeyPoints {
    std::vector<cv::Point3f> pts;   // COCO17: x, y, conf
};

// ---- PaddleDetection PP-YOLOE detector wrapper (MOT / smoking) ----
// Preprocess (verified on engine): direct resize to NxN, BGR 0-255 raw
// (smoking variant: ImageNet-normalized RGB), scale_factor=[N/h, N/w].
// Output boxes are already in original frame coords: [cls, score, x1,y1,x2,y2].
class PpDetModel {
public:
    bool load(const std::string &enginePath, bool imagenetNorm);
    // idempotent multi-channel load: shared instance loads exactly once
    bool loadOnce(const std::string &enginePath, bool imagenetNorm);
    bool ready() const { return eng_.ready(); }
    std::vector<DetResult> infer(const cv::Mat &bgr, float scoreThresh);
    int inputSize() const { return size_; }

    // process-wide shared instances (stateless models: MOT / smoking).
    // 16 channels sharing one engine = 1.5 GB VRAM instead of 24 GB;
    // inference is serialized by the model mutex + global GPU lock
    static PpDetModel &sharedMot();
    static PpDetModel &sharedSmoke();

private:
    TrtEngine eng_;
    bool norm_ = false;
    int size_ = 640;
    std::mutex mtx_;   // full infer transaction (preprocess -> nms)
    bool loadUnlocked(const std::string &enginePath, bool imagenetNorm);
    // reused buffers (per-frame path: no heap churn)
    cv::Mat im_;
    std::vector<float> input_, boxes_;
};

// ---- HRNet keypoint wrapper (top-down, per-person crop 256x192) ----
// Industrial batched inference (official PP-Human batch_size=16): all person
// crops go to the GPU in ONE [n,3,256,192] pass instead of n separate calls.
class KptModel {
public:
    bool load(const std::string &enginePath);
    bool loadOnce(const std::string &enginePath);
    bool ready() const { return eng_.ready(); }
    // crop: person bbox regions from original frame; out aligned with crops
    bool inferBatch(const cv::Mat &bgr, const std::vector<cv::Rect2f> &bboxes,
                    std::vector<std::vector<cv::Point3f>> &out);
    // process-wide shared instance (stateless batched keypoints)
    static KptModel &shared();
private:
    TrtEngine eng_;
    std::mutex mtx_;   // full infer transaction (crop -> heatmap -> decode)
    // reused buffers (high-frequency path: no per-call heap churn)
    cv::Mat crop_;
    std::vector<float> input_, heat_;
};

// ---- shared TRT engine for per-channel stateful action models ----
// fight/fall keep per-channel state (clip / keypoint windows), but deserializing
// their engines on EVERY source switch is expensive (ppTSM 53MB, STGCN 6.6MB)
// and was the cause of multi-second freezes when switching cameras.
// Load each engine ONCE process-wide; mtx serializes the full
// setInput -> infer -> getOutput transaction across channels (device buffers
// live inside TrtEngine and must not be overwritten by a concurrent channel).
struct SharedActionEngine {
    TrtEngine eng;
    std::mutex mtx;
    static SharedActionEngine &fight();
    static SharedActionEngine &fall();
    bool loadOnce(const std::string &path);
};

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

// ---- STGCN skeleton classifier (falling, rolling 50-frame keypoint window) ----
// Industrial sliding-window variant of the official KeyPointBuff pipeline:
// keeps the latest 50 KPT frames per track and infers EVERY KPT frame
// (10 Hz, ~1.4 ms per inference) instead of the official
// "wait 50 frames, infer once, wipe buffer" (5 s blind delay).
// Coords normalized per-frame to person box scaled to 384x512 (official
// parse_mot_keypoint); AutoPadding-style zero head padding for short windows.
// Input [1,2,50,17,1]; logits [1,2]; argmax==0 means Falling.
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
