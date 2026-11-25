#pragma once
#include <QThread>
#include <QDateTime>
#include <QCoreApplication>
#include <QDir>
#include <atomic>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "Config.h"
#include "PPHuman.h"
#include "ByteTracker.h"
#include "BehaviorDetector.h"

Q_DECLARE_METATYPE(cv::Mat)   // queued cross-thread signal payload

// PP-Human 行为监测工作线程（工业流水线）：
// 抓帧 → MOT检测+跟踪 → 语义分频(KPT等) → 规则行为 → 信号槽发主线程
class VideoWorker : public QThread {
    Q_OBJECT
public:
    explicit VideoWorker(QObject *parent = nullptr);
    ~VideoWorker() override;

    void init(const Config &cfg);
    void stop();
    // Register/unregister the live ffmpeg process handle for clean shutdown.
    void registerChild(void *hProc);
    void killChild();
    bool isStopping() const { return stopFlag_.load(); }  // pipe read watchdog
    // runtime-adjustable model params (callable from UI thread anytime)
    void setRuntimeParams(const Config &cfg);
    // idempotent engine load for late-enabled behaviors (hot settings)
    void loadIfNeeded(const Config &cfg);

    // hot-adjustable scalars (benign cross-thread writes of aligned values)
    std::atomic<float> smokeInterval_{0.5f};   // seconds between smoke checks
    std::atomic<int>   fallKeepFrames_{80};    // FALL alert display frames
    std::atomic<int>   minPersonH_{80};        // min person box height for action AI

signals:
    // frame rendered with overlays; stats for status bar
    void frameReady(const cv::Mat &frame, double fps, int frameCount,
                    double inferMs, int detCount);
    // behavior alert (rising edge): type name + detail
    void behaviorAlert(const QString &type, const QString &detail);
    // cannot open video source / engine load failure
    void sourceError(const QString &msg);
    void finished();

protected:
    void run() override;

private:
    void drawTracks(cv::Mat &frame, const std::vector<Trk> &tracks);
    void drawOverlay(cv::Mat &frame, double inferMs, double fps);
    // process-isolated decode for rtsp:// (OpenCV CAP_FFMPEG segfaults on
    // long rtsp sessions and asserts on rebuild — ffmpeg child process is
    // the industrial fix: crash-isolated, self-restarting, session rebuilt)
    void pipeLoop();
    // alert pipeline (NVR standard): evidence snapshot + timestamped trace + signal
    void logAlert(const cv::Mat &frame, const std::vector<Trk> &tracks,
                  const QString &type, const QString &detail);
    // decode thread loop: capture runs ahead of inference (industrial
    // pipeline: decode and infer overlap; realtime sources drop stale
    // frames, file sources use bounded backpressure)
    void capLoop();

    Config cfg_;
    cv::VideoCapture cap_;
    std::string channel_;   // short source tag for logs / evidence files
    double lastUiMs_ = -1e9;   // UI emit rate limit (~12 fps display)

    // runtime-adjustable (settings dialog works while running):
    std::atomic<int> enabledMask_ {0x7F};      // behavior switch bitmask
    std::atomic<float> thSmoke_ {0.65f};
    std::atomic<float> thFight_ {0.50f};
    std::atomic<float> thFall_ {0.50f};
    static constexpr int BIT_CROWD = 1 << 0, BIT_LOITER = 1 << 1,
        BIT_RUN = 1 << 2, BIT_POSTURE = 1 << 3, BIT_SMOKE = 1 << 4,
        BIT_FIGHT = 1 << 5, BIT_FALL = 1 << 6;
    // on-frame alert banner (latest alert shown for 5 s, industrial overlay)
    std::mutex alertBannerMtx_;
    std::string latestAlert_;
    double latestAlertMs_ = -1e9;

    // stateless engines are process-wide shared (multi-channel VRAM):
    // MOT / smoking / KPT loadOnce and serialize internally per model;
    // fight & fall keep per-channel state (clip / keypoint windows)

    // capture thread state
    std::thread capThread_;
    std::mutex qMtx_;
    std::condition_variable qCv_;
    std::deque<cv::Mat> q_;
    bool srcOpened_ = false;
    bool srcEnded_ = false;
    std::string srcErr_;
    double srcFps_ = 25.0;
    int srcW_ = 0, srcH_ = 0;

    // stateless engines are process-wide shared (multi-channel VRAM):
    // MOT / smoking / KPT loadOnce and serialize internally per model;
    // fight & fall keep per-channel state (clip / keypoint windows)
    PpDetModel &mot_;
    PpDetModel &smoke_;
    KptModel &kpt_;
    FightModel fight_;   // per-channel clip state
    FallModel fall_;     // per-channel keypoint windows
    ByteTracker tracker_;
    BehaviorDetector behavior_;

    // per-instance pipeline state — NO statics here: every channel must be
    // independently restartable and safe to run side by side (multi-channel)
    std::unordered_map<int, std::vector<cv::Point3f>> kptById_;
    double lastKpt_ = -1;                              // last analysed frame KPT ran on (every frame, official PP-Human)
    double lastSmoke_ = -1;                            // smoke 2 Hz gate
    std::unordered_map<int, bool> smokeState_;
    std::unordered_map<int, int> smokeLife_;   // marker lifetime (display_frames: 80)
    // smoke debouncing: consecutive positive checks per track before alerting
    std::unordered_map<int, int> smokeStreak_;
    std::unordered_map<int, double> smokeLastAlert_;  // video-time of last SMOKE alert (cooldown)

    // fight (ppTSM): latest clip decision, kept until next clip
    bool fightState_ = false;
    float fightScore_ = 0.f;
    // fight debouncing: consecutive positive clips before alerting
    int fightStreak_ = 0;
    bool fightReported_ = false;
    // fall (STGCN): sticky display life per track (frames, official
    // ActionVisualHelper semantics: state holds, non-falling results ignored)
    std::unordered_map<int, int> fallLife_;
    std::unordered_set<int> fallReported_;

    std::string alertDir_;   // evidence snapshots (exe_dir/alerts)

    int frameCount_ = 0;
    std::atomic_bool stopFlag_{false};
    // ffmpeg child process (owned here so stop()/destructor can kill it even
    // when the capture thread is blocked in ReadFile on the pipe)
    std::mutex childMtx_;
    void *childProc_ = nullptr;   // HANDLE, void* to keep WinAPI out of header
};
