#pragma once
#include <string>
#include <algorithm>

struct Config {
    std::string video_path;       // file path / camera index digit / rtsp url
    std::string mot_engine_path;     // person detection (ppyoloe)
    std::string kpt_engine_path;     // HRNet keypoints (optional)
    std::string smoke_engine_path;   // smoking detector (optional)
    std::string fight_engine_path;   // ppTSM (optional, phase 3)
    std::string fall_engine_path;    // STGCN (optional, phase 3)

    // behavior switches (UI checkboxes)
    bool beh_crowd   = true;   // crowd gathering (rule)
    bool beh_loiter  = true;   // loitering (rule)
    bool beh_run     = true;   // running (rule)
    bool beh_posture = true;   // abnormal posture (rule, needs kpt)
    bool beh_smoke   = true;   // smoking (detector)
    bool beh_fight   = true;   // fight (ppTSM)
    bool beh_fall    = true;   // falling (STGCN, needs kpt)

    // per-behavior confidence thresholds (UI spinboxes, one per row)
    // rule rows: official crop_thresh 0.5 — minimum detection score
    // participating in that behavior (infer_cfg_pphuman.yml)
    float th_crowd   = 0.50f;
    float th_loiter  = 0.50f;
    float th_run     = 0.50f;
    float th_posture = 0.50f;
    // smoke: official ID_BASED_DETACTION threshold 0.6; local 0.65 is stricter
    // fight/fall: official action_infer threshold 0.5
    float th_smoke   = 0.65f;
    float th_fight   = 0.50f;
    // STGCN false falls on non-fall clips are weak (softmax ~0.79-0.85),
    // real falls are near 1.00; 0.85 separates them cleanly.
    float th_fall    = 0.85f;

    // per-model tuning parameters ("设置" dialogs)
    int   crowdCount   = 5;      // persons to trigger crowd
    float crowdSec     = 3.0f;   // sustained seconds
    float loiterSec    = 10.0f;  // dwell seconds
    float loiterMove   = 50.0f;  // net displacement px
    float runSpeed     = 400.0f; // px/s
    int   runMinFrames = 5;
    float postureAngle = 60.0f;  // torso angle deg
    float postureSym   = 0.30f;  // limb symmetry ratio
    float smokeInterval = 0.5f;  // seconds between smoke checks
    int   fightSampleFreq = 7;   // sample every N frames
    int   fallKeepFrames  = 80;  // alert display frames (official display_frames)
    int   analyzeEvery    = 1;   // analyse every Nth frame (multi-channel
                                 // deployment: 15fps cam + N=2 -> 7.5fps/路)

    // Far-scene guard: FALL/SMOKE/FIGHT need enough person pixels to be
    // reliable. Tracks shorter than this height (px) are skipped for those
    // action classifiers; CROWD/LOITER/RUN rules are unaffected.
    // 0 = disabled (classify everyone). Typical wide plaza cam: 80~120.
    int   minPersonH      = 80;

    bool needKpt() const { return beh_posture || beh_fall; }
    // MOT detection threshold, fixed to the official value (PP-Human v2.9
    // tracker_config.yml: BOTSORT track_high_thresh 0.3 / BYTE det_thresh 0.3).
    // Per-behavior filtering uses the rule-row thresholds (crop_thresh 0.5)
    float motThresh() const { return 0.30f; }
};
