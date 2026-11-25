#pragma once
#include "TrtEngine.h"
#include <mutex>
#include <string>

// ---- shared TRT engine for per-channel stateful action models ----
// Fight/fall keep per-channel state (clip / keypoint windows), but deserializing
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
