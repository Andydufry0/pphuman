#include "VideoWorker.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <QString>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>

// SkylineWebcams 公共摄像头（广场/街道）的 HLS token 是短命的，
// 每次开流前抓一次摄像头页面，解析出当前 token 再交给 ffmpeg。
// 页面地址是稳定的，token 只在本次会话有效。
// httpStatus 输出 HTTP 状态码（429=被限流），供调用方做退避，避免疯狂重试
// 把源彻底打封（实测每 2 秒抓一次页面几分钟后该 IP 被 429）。
static std::string resolveSkylineStream(const std::string &pageUrl,
                                        int *httpStatus = nullptr) {
    if (httpStatus) *httpStatus = 0;
    const std::string prefix = "https://";
    if (pageUrl.rfind(prefix, 0) != 0) return std::string();
    size_t slash = pageUrl.find('/', prefix.size());
    if (slash == std::string::npos) return std::string();
    std::wstring host(pageUrl.begin() + prefix.size(), pageUrl.begin() + slash);
    std::wstring path(pageUrl.begin() + slash, pageUrl.end());

    std::string token;
    // 用浏览器 UA：实测自定义 UA 更容易被 Skyline 限流(429)
    HINTERNET sess = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                 WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                 WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!sess) return token;
    HINTERNET conn = WinHttpConnect(sess, host.c_str(), INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (conn) {
        HINTERNET req = WinHttpOpenRequest(conn, L"GET", path.c_str(), nullptr,
                                           WINHTTP_NO_REFERER,
                                           WINHTTP_DEFAULT_ACCEPT_TYPES,
                                           WINHTTP_FLAG_SECURE);
        if (req && WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                      nullptr, 0, 0, 0) &&
            WinHttpReceiveResponse(req, nullptr)) {
            // 取 HTTP 状态码：429 = 限流，调用方需长退避
            if (httpStatus) {
                DWORD code = 0, len = sizeof(code);
                WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE |
                                    WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                                    &code, &len, WINHTTP_NO_HEADER_INDEX);
                *httpStatus = (int)code;
            }
            std::string html;
            char buf[8192];
            DWORD got = 0;
            while (WinHttpReadData(req, buf, sizeof(buf), &got) && got > 0)
                html.append(buf, got);
            const char *key = "livee.m3u8?a=";
            size_t p = html.find(key);
            if (p != std::string::npos) {
                p += strlen(key);
                size_t e = html.find_first_of("\"'", p);
                if (e != std::string::npos) token = html.substr(p, e - p);
            }
        }
        if (req) WinHttpCloseHandle(req);
        WinHttpCloseHandle(conn);
    }
    WinHttpCloseHandle(sess);
    if (token.empty()) return std::string();
    return "https://hd-auth.skylinewebcams.com/live.m3u8?a=" + token;
}

// 无窗口启动 ffmpeg 并读它的 stdout：GUI 程序里 _popen 会弹 cmd 黑框，
// 用 CreateProcess + CREATE_NO_WINDOW + 匿名管道彻底隐藏控制台。
struct FfmpegPipe {
    HANDLE hProc = nullptr;
    HANDLE hThread = nullptr;
    HANDLE hRead = nullptr;
    bool ok = false;
    VideoWorker *owner = nullptr;    // registers hProc for stop()-time kill

    explicit FfmpegPipe(const std::string &cmd, VideoWorker *ow = nullptr) : owner(ow) {
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;
        HANDLE hWrite = nullptr;
        // 4MB 缓冲，容纳 720p 单帧（~2.7MB）避免管道写阻塞
        if (!CreatePipe(&hRead, &hWrite, &sa, 4 * 1024 * 1024)) {
            hRead = nullptr;
            return;
        }
        // 只让子进程继承写端，读端留在父进程
        SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

        // stderr / stdin 指向 NUL，避免无效句柄让 ffmpeg 异常退出，
        // 同时保证日志不会混进 stdout 的 rawvideo 数据
        HANDLE hNul = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

        STARTUPINFOA si;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = hWrite;
        si.hStdError = hNul;
        si.hStdInput = hNul;

        PROCESS_INFORMATION pi;
        ZeroMemory(&pi, sizeof(pi));
        // CreateProcess 会原地改命令行缓冲区，必须给可写副本
        std::string mutableCmd = cmd;
        ok = CreateProcessA(nullptr, &mutableCmd[0], nullptr, nullptr, TRUE,
                            CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        CloseHandle(hWrite);   // 父进程不需要写端；子进程已继承
        if (hNul) CloseHandle(hNul);
        if (!ok) {
            CloseHandle(hRead);
            hRead = nullptr;
            return;
        }
        hProc = pi.hProcess;
        hThread = pi.hThread;
        if (owner) owner->registerChild(hProc);
    }

    // 带超时的定长读一帧：管道可能分段返回，循环拼满 frameBytes 才返回。
    // 关键（工业实时流）：绝不能无限阻塞在 ReadFile。用 PeekNamedPipe + 短
    // sleep 轮询，一旦子进程退出或超过 stallMs 没有任何字节到达就立即返回，
    // 让 pipeLoop 杀掉卡死的 ffmpeg 并重新解析 token 重建会话。这样即使
    // HLS token 过期、ffmpeg 卡在拉流重试不死不吐，画面也不会永久定格。
    size_t readFrame(uint8_t *dst, size_t frameBytes, DWORD stallMs = 10000) {
        size_t total = 0;
        auto lastData = std::chrono::steady_clock::now();
        while (total < frameBytes) {
            // 管道当前可读字节数
            DWORD avail = 0;
            if (!PeekNamedPipe(hRead, nullptr, 0, nullptr, &avail, nullptr))
                break;                       // 管道失效
            if (avail > 0) {
                DWORD want = (DWORD)std::min<size_t>(avail, frameBytes - total);
                DWORD got = 0;
                if (!ReadFile(hRead, dst + total, want, &got, nullptr) || got == 0)
                    break;
                total += got;
                lastData = std::chrono::steady_clock::now();
                continue;
            }
            // 管道空：子进程已退出 -> 不会有新数据了
            if (hProc && WaitForSingleObject(hProc, 0) == WAIT_OBJECT_0)
                break;
            // 上游停采或用户 stop -> 立即退出，不阻塞线程结束
            if (owner && owner->isStopping())
                break;
            // 停滞看门狗：长时间零字节判定流卡死
            if (std::chrono::steady_clock::now() - lastData >
                    std::chrono::milliseconds(stallMs))
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return total;
    }

    ~FfmpegPipe() {
        if (hProc && owner) owner->registerChild(nullptr);  // unregister first
        if (hRead) { CloseHandle(hRead); hRead = nullptr; }
        if (hProc) {
            // killChild() may already have terminated it; still ensure exit
            if (WaitForSingleObject(hProc, 300) == WAIT_TIMEOUT)
                TerminateProcess(hProc, 0);
            CloseHandle(hProc);
        }
        if (hThread) CloseHandle(hThread);
    }
};

// 本地摄像头索引（全数字字符串，如 "0"/"12"）-> true；文件路径/network -> false
static bool isCameraIndex(const std::string &s) {
    if (s.empty()) return false;
    for (char c : s) if (!isdigit((unsigned char)c)) return false;
    return true;
}

// 本地摄像头默认跑最大分辨率。
// OpenCV 对不支持的分辨率不报错（MSMF/DSHOW 要么静默忽略、要么就近给一档），
// 所以只能"设一次、读回一次"：逐档试，记住实际拿到过的最大尺寸，最后落回该档。
// 先切 MJPG：多数 UVC 摄像头 2K/4K 档只在 MJPEG 下可用，YUY2 带宽不够；
// 设置失败无副作用（读回仍是原格式）。
static void applyMaxCameraResolution(cv::VideoCapture &cap) {
    static const int cand[][2] = {
        {3840,2160},{2592,1944},{2560,1440},{2560,1920},{2048,1536},
        {1920,1080},{1600,1200},{1280,960},{1280,720},{1024,768},{800,600},{640,480}
    };
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    int bestW = 0, bestH = 0;
    for (const auto &c : cand) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  c[0]);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, c[1]);
        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        if ((long long)w * h > (long long)bestW * bestH) { bestW = w; bestH = h; }
        if (w >= c[0] && h >= c[1]) break;   // 该档确实拿到了，往下只会更小
    }
    if (bestW > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  bestW);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, bestH);
    }
    fprintf(stderr, "[camera] resolution -> %dx%d\n", bestW, bestH);
    fflush(stderr);
}

static cv::Scalar trackColor(int id) {
    return cv::Scalar((id * 47) % 256, (id * 91) % 256, (id * 173) % 256);
}

VideoWorker::VideoWorker(QObject *parent)
    : QThread(parent),
      mot_(PpDetModel::sharedMot()),
      smoke_(PpDetModel::sharedSmoke()),
      kpt_(KptModel::shared()) {}

VideoWorker::~VideoWorker() { stop(); wait(10000); }

void VideoWorker::init(const Config &cfg) {
    stopFlag_ = false;
    frameCount_ = 0;
    cfg_ = cfg;
    setRuntimeParams(cfg);   // sync runtime switches/thresholds at start

    // channel tag: last path segment without extension (rtsp url -> tail)
    size_t p = cfg_.video_path.find_last_of("/\\");
    std::string tail = p == std::string::npos ? cfg_.video_path
                                              : cfg_.video_path.substr(p + 1);
    size_t e = tail.rfind('.');
    if (e != std::string::npos && e > 0) tail = tail.substr(0, e);
    channel_ = tail;
}

void VideoWorker::registerChild(void *hProc) {
    std::lock_guard<std::mutex> lk(childMtx_);
    childProc_ = hProc;
}

void VideoWorker::killChild() {
    HANDLE h = nullptr;
    {
        std::lock_guard<std::mutex> lk(childMtx_);
        h = (HANDLE)childProc_;   // 只 Terminate，句柄仍归 FfmpegPipe 所有，
                                 // 由它析构时 CloseHandle，避免重复关闭
    }
    if (h) TerminateProcess(h, 0);   // 管道写端断开 -> 阻塞的 ReadFile 立即返回
}

void VideoWorker::stop() {
    stopFlag_ = true;
    qCv_.notify_all();   // wake run() / capLoop() waits
    killChild();         // unblock pipe read, kill ffmpeg (no orphan)
}

void VideoWorker::logAlert(const cv::Mat &frame, const std::vector<Trk> &tracks,
                           const QString &type, const QString &detail) {
    // evidence snapshot: annotated frame saved to alerts/ (NVR standard,
    // required for post-event review); one file per rising edge only
    if (!alertDir_.empty() && !frame.empty()) {
        cv::Mat ev = frame.clone();
        drawTracks(ev, tracks);
        QString ch = QString::fromStdString(channel_);
        QString stamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_zzz");
        QString path = QString::fromStdString(alertDir_) + "/" + stamp
                       + (ch.isEmpty() ? "" : "_" + ch) + "_" + type + ".jpg";
        std::vector<int> p = {cv::IMWRITE_JPEG_QUALITY, 85};
        cv::imwrite(path.toLocal8Bit().toStdString(), ev, p);
    }
    fprintf(stderr, "[%s] [%s] [ALERT] %s %s\n",
            QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
                .toUtf8().constData(),
            channel_.c_str(),
            type.toUtf8().constData(), detail.toUtf8().constData());
    fflush(stderr);
    emit behaviorAlert(type, detail);
    // on-frame banner: the monitoring view itself carries the alert
    // (headless/oversized grids have no log panel to look at)
    {
        std::lock_guard<std::mutex> lk(alertBannerMtx_);
        latestAlert_ = type.toStdString() + "  " + detail.toStdString();
        latestAlertMs_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
}

void VideoWorker::setRuntimeParams(const Config &cfg) {
    smokeInterval_.store(cfg.smokeInterval);
    fallKeepFrames_.store(cfg.fallKeepFrames);
    minPersonH_.store(cfg.minPersonH);
    fight_.setSampleFreq(cfg.fightSampleFreq);
    // runtime behavior switches + thresholds (settings dialog, hot)
    enabledMask_.store(
        (cfg.beh_crowd   ? BIT_CROWD   : 0) |
        (cfg.beh_loiter  ? BIT_LOITER  : 0) |
        (cfg.beh_run     ? BIT_RUN     : 0) |
        (cfg.beh_posture ? BIT_POSTURE : 0) |
        (cfg.beh_smoke   ? BIT_SMOKE   : 0) |
        (cfg.beh_fight   ? BIT_FIGHT   : 0) |
        (cfg.beh_fall    ? BIT_FALL    : 0));
    thSmoke_.store(cfg.th_smoke);
    thFight_.store(cfg.th_fight);
    thFall_.store(cfg.th_fall);
    // benign cross-thread writes of simple scalars (read each frame)
    behavior_.crowdThreshold_ = cfg.crowdCount;
    behavior_.crowdAlertSec_  = cfg.crowdSec;
    behavior_.loiterSec_      = cfg.loiterSec;
    behavior_.loiterMovePx_   = cfg.loiterMove;
    behavior_.runSpeedPx_     = cfg.runSpeed;
    behavior_.runMinFrames_   = cfg.runMinFrames;
    behavior_.fallAngleDeg_   = cfg.postureAngle;
    behavior_.symmetryThresh_ = cfg.postureSym;
    behavior_.minScoreCrowd_   = cfg.th_crowd;
    behavior_.minScoreLoiter_  = cfg.th_loiter;
    behavior_.minScoreRun_     = cfg.th_run;
    behavior_.minScorePosture_ = cfg.th_posture;
}

// late-enabled behavior engines load on demand (hot settings dialog)
void VideoWorker::loadIfNeeded(const Config &cfg) {
    if (cfg.beh_fall && !fall_.ready())
        fall_.load(cfg.fall_engine_path);
    if (cfg.beh_smoke && !smoke_.ready())
        smoke_.loadOnce(cfg.smoke_engine_path, true);
    if (cfg.needKpt() && !kpt_.ready())
        kpt_.loadOnce(cfg.kpt_engine_path);
    if (cfg.beh_fight && !fight_.ready())
        fight_.load(cfg.fight_engine_path);
}

void VideoWorker::run() {
    // reset all per-run state: this worker must be restartable (stop -> start)
    // and multiple instances must never share state (multi-channel ready)
    q_.clear();
    srcOpened_ = srcEnded_ = false;
    srcErr_.clear();
    kptById_.clear();
    lastKpt_ = lastSmoke_ = -1;
    smokeState_.clear();
    smokeLife_.clear();
    smokeStreak_.clear();
    smokeLastAlert_.clear();
    fallReported_.clear();
    fallLife_.clear();
    fightState_ = false;
    fightScore_ = 0.f;
    behavior_.reset();

    // evidence snapshot directory (NVR standard: every alert leaves a file)
    alertDir_ = QCoreApplication::applicationDirPath().toLocal8Bit().toStdString()
                + "/alerts";
    QDir().mkpath(QString::fromStdString(alertDir_));

    // Load engines in worker thread (CUDA context is thread-local)
    auto engT0 = std::chrono::steady_clock::now();
    if (!mot_.loadOnce(cfg_.mot_engine_path, false)) {
        emit sourceError("MOT engine load failed: " +
                         QString::fromStdString(cfg_.mot_engine_path));
        emit finished(); return;
    }
    // load engines only for enabled behaviors (official PP-Human builds
    // modules on demand; unchecked models never occupy GPU memory).
    // mot/smoke/kpt are process-wide shared: loadOnce is idempotent —
    // 16 channels load them exactly once (DeepStream single-engine model)
    // NOTE: load selection uses the init-time cfg_, NOT the runtime mask —
    // the mask starts fully open and is a runtime switch only
    if (cfg_.beh_fall)
        fall_.load(cfg_.fall_engine_path);           // per-channel state
    if (cfg_.beh_smoke)
        smoke_.loadOnce(cfg_.smoke_engine_path, true);   // shared, fail-soft
    if (cfg_.needKpt())                              // posture || fall
        kpt_.loadOnce(cfg_.kpt_engine_path);             // shared, fail-soft
    if (cfg_.beh_fight)
        fight_.load(cfg_.fight_engine_path);         // per-channel state
    fprintf(stderr, "[switch] engines ready in %.0f ms\n",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - engT0).count());
    fflush(stderr);

    // open source: file / camera index / rtsp url (decode thread)
    capThread_ = std::thread([this] { capLoop(); });
    {
        std::unique_lock<std::mutex> lk(qMtx_);
        qCv_.wait(lk, [this] { return srcOpened_; });
        if (!srcErr_.empty()) {
            emit sourceError(QString::fromStdString(srcErr_));
            emit finished();
            capThread_.join();
            return;
        }
    }
    double fps = srcFps_ / std::max(1, cfg_.analyzeEvery);
    behavior_.setFps(fps);
    setRuntimeParams(cfg_);   // per-model tuning params
    std::cout << "[VideoWorker] source " << (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH)
              << "x" << (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT)
              << " @ " << fps << " fps\n";

    auto tStart = std::chrono::high_resolution_clock::now();
    auto tFps = tStart;
    int fpsCnt = 0;
    double fpsNow = 0.0;
    int rawIdx = 0;   // source frame index (pre-skip)

    while (!stopFlag_) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(qMtx_);
            qCv_.wait(lk, [this] { return !q_.empty() || srcEnded_ || stopFlag_; });
            if (q_.empty()) break;   // source ended and drained
            frame = std::move(q_.front());
            q_.pop_front();
        }
        qCv_.notify_one();           // release file-source backpressure
        rawIdx++;
        // frame skipping (multi-channel): analyse every Nth source frame;
        // skipped frames do not advance frameCount_ so all downstream time
        // math (10 Hz kpt gate, px/s speeds, dwell seconds) stays consistent
        if (cfg_.analyzeEvery > 1 && (rawIdx - 1) % cfg_.analyzeEvery != 0)
            continue;
        frameCount_++;
        fpsCnt++;

        auto t0 = std::chrono::high_resolution_clock::now();


        // 1. person detection + tracking (loosest rule threshold; per-behavior
        //    score filtering happens in BehaviorDetector)
        auto dets = mot_.infer(frame, cfg_.motThresh());
        std::vector<cv::Rect2f> detBoxes;
        std::vector<float> detScores;
        for (auto &d : dets) {
            detBoxes.push_back(d.box);
            detScores.push_back(d.score);
        }
        auto tracks = tracker_.update(detBoxes, detScores);

        // 2. keypoint branch: every analysed frame (posture / fall need it)
        // cache by track_id: track order changes per frame, key by ID
        if (kpt_.ready() && (enabledMask_ & (BIT_POSTURE | BIT_FALL))) {
            // KPT 每个分析帧都跑一次 —— 官方 PP-Human 就是这么做的（每帧
            // crop_image_with_mot -> HRNet，没有分频门控）。关键点在于
            // STGCN 的 max_frames=50 指的是 50 个**连续视频帧**，25fps 下
            // ≈ 2 s，和训练集帧率一致。原实现按 10 Hz 采样，50 帧 = 5 s，
            // 时间轴被拉慢 2.5 倍：摔倒的快速下坠被摊平成慢动作、静止画面
            // 又被拉长，结果是既容易误报、又需要更长的连续确认才报。
            if (lastKpt_ != (double)frameCount_) {
                lastKpt_ = (double)frameCount_;
                // official PP-Human KPT batch_size: 16 persons per pass
                int n = std::min<size_t>(tracks.size(), 16);
                std::vector<std::vector<cv::Point3f>> kptsAll;
                std::vector<TrackBox> kb;
                std::vector<cv::Rect2f> fboxEx;   // expanded crops for STGCN
                std::vector<cv::Rect2f> crops;    // KPT batch input (parallel to kb)
                crops.reserve(n);
                for (int i = 0; i < n; ++i) {
                    cv::Rect2f b = tracks[i].tlwh;
                    // far-scene guard: tiny persons give unstable keypoints
                    // -> STGCN false falls. Skip them (0 = disabled).
                    int minH = minPersonH_.load();
                    if (minH > 0 && b.height < (float)minH) continue;
                    // official crop_thresh 0.5: KPT / skeleton action only run
                    // on detections with score >= 0.5 (MOT tracks down to 0.3).
                    // Low-score boxes give garbage keypoints -> STGCN noise.
                    if (tracks[i].score < 0.5f) continue;
                    // official KPT branch crops with 30%-expanded box (expand_crop)
                    float hh = b.height * 1.3f * 0.5f, wh = b.width * 1.3f * 0.5f;
                    if (hh > wh * 4.f / 3.f) wh = hh * 0.75f;
                    float cx = b.x + b.width * 0.5f, cy = b.y + b.height * 0.5f;
                    cv::Rect2f crop(cx - wh, cy - hh, 2 * wh, 2 * hh);
                    crop &= cv::Rect2f(0, 0, (float)frame.cols, (float)frame.rows);
                    if (crop.width < 8 || crop.height < 8) continue;
                    crops.push_back(crop);
                    TrackBox tb;
                    tb.id = tracks[i].id;
                    tb.x = b.x; tb.y = b.y; tb.w = b.width; tb.h = b.height;
                    tb.score = tracks[i].score;
                    kb.push_back(tb);
                    // STGCN normalization bbox MUST be the original MOT box,
                    // NOT the expanded crop. Keypoints are transformed back to
                    // full-frame coords (see PPHuman.cpp inferBatch), and the
                    // official parse_mot_keypoint uses kpt_seq.bboxes (raw MOT
                    // box) to normalize. Mixing crop box with full-frame kpts
                    // produced garbage coords -> STGCN always "Falling" (1.00)
                    // even on non-fall videos.
                    fboxEx.push_back(b);   // raw track box [x,y,w,h], x2y2 derived in FallModel
                }
                // batched GPU pass (official batch_size): n persons in ONE infer
                if (!crops.empty() && kpt_.inferBatch(frame, crops, kptsAll)) {
                    for (int i = (int)kb.size() - 1; i >= 0; --i) {
                        if (!kptsAll[i].empty()) continue;
                        kb.erase(kb.begin() + i);
                        fboxEx.erase(fboxEx.begin() + i);
                        kptsAll.erase(kptsAll.begin() + i);
                    }
                } else {
                    kb.clear(); fboxEx.clear(); kptsAll.clear();
                }
                for (size_t i = 0; i < kb.size(); ++i)
                    kptById_[kb[i].id] = kptsAll[i];

                if ((enabledMask_ & BIT_POSTURE) && !kptsAll.empty())
                    behavior_.updateKeypoints(kptsAll, kb);

                // falling (STGCN): official semantics — one non-overlapping
                // 50-frame window per decision (KeyPointBuff pops and resets),
                // plus ActionVisualHelper's sticky state: while a track is in
                // the falling state, later non-falling results do NOT override
                // it until the display window expires. That sticky state is
                // what keeps score jitter around the threshold from re-firing.
                if (fall_.ready() && (enabledMask_ & BIT_FALL)) {
                    std::vector<int> fids;
                    fids.reserve(kb.size());
                    for (auto &t : kb) fids.push_back(t.id);
                    auto fres = fall_.update(fids, fboxEx, kptsAll);
                    for (auto &r : fres) {
                        bool fall = r.falling && r.score >= thFall_.load();
                        if (!fall) continue;              // sticky: ignore false
                        // fall (falling) is the only action class in the
                        // official STGCN model (0=fall, 1=other) — there is no
                        // "get up" class. Detect the transition here instead.
                        // Scene gate (NOT official): STGCN keeps scoring
                        // ~0.96-1.00 on a person standing still on the phone.
                        // At a real fall keypoints span the box vertically
                        // (vspan>0.65) and the box is either still tall
                        // (w/h<0.9, fall in progress) OR the torso has already
                        // gone near-horizontal (torso<40 deg, lying flat).
                        // A standing subject fails both branches.
                        bool geomOk = false;
                        for (size_t bi = 0; bi < kb.size(); ++bi) {
                            if (kb[bi].id != r.id || bi >= kptsAll.size()) continue;
                            float w = std::max(kb[bi].w, 1e-3f), h = std::max(kb[bi].h, 1e-3f);
                            float ymin = 1e9f, ymax = -1e9f;
                            for (auto &p : kptsAll[bi]) {
                                if (p.z < 0.2f) continue;
                                ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
                            }
                            float vs = (ymax > ymin) ? (ymax - ymin) / h : 0.f;
                            float sx = (kptsAll[bi][5].x + kptsAll[bi][6].x) * .5f;
                            float sy = (kptsAll[bi][5].y + kptsAll[bi][6].y) * .5f;
                            float hx = (kptsAll[bi][11].x + kptsAll[bi][12].x) * .5f;
                            float hy = (kptsAll[bi][11].y + kptsAll[bi][12].y) * .5f;
                            float deg = std::atan2(std::fabs(hy - sy), std::fabs(hx - sx)) * 57.2958f;
                            geomOk = (vs > 0.65f) && ((w / h < 0.90f) || (deg < 40.0f));
                            fprintf(stderr, "[FALLCHK] id=%d score=%.3f wh=%.2f vspan=%.2f torso=%.1f gate=%d\n",
                                    r.id, r.score, w / h, vs, deg, (int)geomOk);
                            break;
                        }
                        if (!geomOk) continue;
                        bool inState = fallLife_.count(r.id) > 0;
                        fallLife_[r.id] = fallKeepFrames_.load();   // refresh / enter
                        if (!inState && !fallReported_.count(r.id)) {
                            logAlert(frame, tracks, "FALL",
                                QString("id=%1 score=%2").arg(r.id).arg(r.score, 0, 'f', 2));
                            fallReported_.insert(r.id);
                        }
                    }
                }

                // drop cache of vanished ids
                for (auto it = kptById_.begin(); it != kptById_.end();) {
                    bool found = false;
                    for (auto &t : tracks) if (t.id == it->first) { found = true; break; }
                    if (!found) it = kptById_.erase(it); else ++it;
                }
                for (auto it = fallReported_.begin(); it != fallReported_.end();) {
                    bool found = false;
                    for (auto &t : tracks) if (t.id == *it) { found = true; break; }
                    if (!found) it = fallReported_.erase(it); else ++it;
                }
                // drop smoke/smoke-state of vanished ids (avoid id-reuse carry-over)
                for (auto it = smokeStreak_.begin(); it != smokeStreak_.end();) {
                    bool found = false;
                    for (auto &t : tracks) if (t.id == it->first) { found = true; break; }
                    if (!found) it = smokeStreak_.erase(it); else ++it;
                }
                for (auto it = smokeLastAlert_.begin(); it != smokeLastAlert_.end();) {
                    bool found = false;
                    for (auto &t : tracks) if (t.id == it->first) { found = true; break; }
                    if (!found) it = smokeLastAlert_.erase(it); else ++it;
                }
                for (auto it = smokeState_.begin(); it != smokeState_.end();) {
                    bool found = false;
                    for (auto &t : tracks) if (t.id == it->first) { found = true; break; }
                    if (!found) it = smokeState_.erase(it); else ++it;
                }
            }
        }

        // 3. rule-based behaviors (every frame, microseconds)
        std::vector<TrackBox> tbs;
        tbs.reserve(tracks.size());
        for (auto &t : tracks) {
            TrackBox tb;
            tb.id = t.id;
            tb.cls = 0;
            tb.score = t.score;
            tb.x = t.tlwh.x; tb.y = t.tlwh.y;
            tb.w = t.tlwh.width; tb.h = t.tlwh.height;
            tbs.push_back(tb);
        }
        auto events = behavior_.update(tbs, frameCount_,
                                       frame.cols, frame.rows);
        static const char *typeName[] = {"?", "CROWD", "LOITER", "RUN", "POSTURE"};
        for (auto &e : events) {
            // respect UI switches
            if ((e.type == BehaviorEvent::CROWD   && !(enabledMask_ & BIT_CROWD)) ||
                (e.type == BehaviorEvent::LOITER  && !(enabledMask_ & BIT_LOITER)) ||
                (e.type == BehaviorEvent::RUN     && !(enabledMask_ & BIT_RUN)) ||
                (e.type == BehaviorEvent::POSTURE && !(enabledMask_ & BIT_POSTURE)))
                continue;
            // official semantics: alert_rise only — one alert per state
            // transition (extended_actions.py), never repeat while active
            if (!e.alert) continue;
            logAlert(frame, tracks, QString(typeName[(int)e.type]),
                     QString::fromStdString(e.detail));
        }

        // 4. smoking branch @ 2 Hz: per-person expanded crop, class0=smoking
        double nowS = frameCount_ / fps;
        if (smoke_.ready() && (enabledMask_ & BIT_SMOKE) && nowS - lastSmoke_ >= smokeInterval_.load()) {
            lastSmoke_ = nowS;
            int n = std::min<size_t>(tracks.size(), 8);
            for (int i = 0; i < n; ++i) {
                // far-scene guard: tiny persons -> smoking detector false positives
                int minH = minPersonH_.load();
                if (minH > 0 && tracks[i].tlwh.height < (float)minH) continue;
                // expand_crop (ratio 0.3), keep aspect if tall
                cv::Rect2f b = tracks[i].tlwh;
                float hh = b.height * 1.3f * 0.5f, wh = b.width * 1.3f * 0.5f;
                if (hh > wh * 4.f / 3.f) wh = hh * 0.75f;
                float cx = b.x + b.width * 0.5f, cy = b.y + b.height * 0.5f;
                cv::Rect crop((int)(cx - wh), (int)(cy - hh),
                              (int)(2 * wh), (int)(2 * hh));
                crop &= cv::Rect(0, 0, frame.cols, frame.rows);
                if (crop.width < 8 || crop.height < 8) continue;
                auto sd = smoke_.infer(frame(crop), 0.0f);
                bool smoking = false;
                for (auto &d : sd)
                    if (d.cls == 0 && d.score >= thSmoke_.load()) { smoking = true; break; }
                smokeState_[tracks[i].id] = smoking;
                if (smoking) {
                    smokeLife_[tracks[i].id] = 80;   // official display_frames
                    // industrial debouncing: single-frame model hits are noise
                    // (official only visualizes, never alerts). Require 3
                    // consecutive checks (~1.5 s @2 Hz) before alerting, and
                    // dedup one alert per track until it recovers.
                    // Cooldown: same target alerts at most once per 20 s of
                    // video time. Without it, a 1-frame detection drop resets
                    // the streak and the same smoker re-alerts every second.
                    auto la = smokeLastAlert_.find(tracks[i].id);
                    bool cooled = (la == smokeLastAlert_.end()) ||
                                  (nowS - la->second >= 20.0);
                    if (++smokeStreak_[tracks[i].id] >= 3 && cooled) {
                        logAlert(frame, tracks, "SMOKE",
                            QString("id=%1").arg(tracks[i].id));
                        smokeLastAlert_[tracks[i].id] = nowS;
                    }
                } else {
                    smokeStreak_[tracks[i].id] = 0;
                }
            }
        }
        // smoking markers with official display_frames: 80 frames (~3.2 s @25fps).
        // Marker stays visible after the alert edge even if the next check is
        // negative (visual continuity, same pattern as FALL fallLife_)
        for (auto it = smokeLife_.begin(); it != smokeLife_.end();) {
            for (const auto &t : tracks) {
                if (t.id != it->first) continue;
                cv::Rect r((int)t.tlwh.x, (int)t.tlwh.y,
                           (int)t.tlwh.width, (int)t.tlwh.height);
                cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 3);
                cv::putText(frame, "SMOKE", cv::Point(r.x, std::max(12, r.y - 4)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
            if (--it->second <= 0) it = smokeLife_.erase(it); else ++it;
        }

        // 5. fight branch (ppTSM): samples internally every 7 frames, infers on
        //    full clip; official VIDEO_ACTION only works when a person is detected
        if (fight_.ready() && (enabledMask_ & BIT_FIGHT) && !tracks.empty()) {
            // far-scene guard: ppTSM on a full frame of tiny people is
            // meaningless — require at least 2 sufficiently large persons
            int minH = minPersonH_.load();
            int big = 0;
            if (minH <= 0) big = (int)tracks.size();
            else for (const auto &t : tracks)
                if (t.tlwh.height >= (float)minH) ++big;
            if (big >= 2) {
                bool fight; float fscore;
                if (fight_.pushFrame(frame, thFight_.load(), fight, fscore)) {
                    // industrial debouncing: one noisy clip is not a fight.
                    // Require 2 consecutive positive clips before alerting,
                    // then dedup until the scene recovers.
                    if (fight && ++fightStreak_ >= 2 &&
                        !fightReported_) {
                        logAlert(frame, tracks, "FIGHT",
                            QString("score=%1").arg(fscore, 0, 'f', 2));
                        fightReported_ = true;
                    } else if (!fight) {
                        fightStreak_ = 0;
                        fightReported_ = false;
                    }
                    fightState_ = fight;
                    fightScore_ = fscore;
                }
            }
        }

        // fall sticky life decays every frame (Python ActionVisualHelper):
        // when a track's state expires, its "already reported" flag is cleared
        // so a genuinely new fall on the same track can alert again.
        for (auto it = fallLife_.begin(); it != fallLife_.end();) {
            if (--it->second <= 0) {
                fallReported_.erase(it->first);
                it = fallLife_.erase(it);
            } else ++it;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double elapsed = std::chrono::duration<double>(t1 - tFps).count();
        if (elapsed >= 1.0) { fpsNow = fpsCnt / elapsed; fpsCnt = 0; tFps = t1; }

        drawTracks(frame, tracks);
        // draw skeletons from id-keyed cache
        {
            static const int PAIRS[][2] = {{15,13},{13,11},{16,14},{14,12},{11,12},
                {5,11},{6,12},{5,6},{5,7},{6,8},{7,9},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6}};
            for (const auto &t : tracks) {
                auto it = kptById_.find(t.id);
                if (it == kptById_.end()) continue;
                const auto &kp = it->second;
                if (kp.size() < 17) continue;
                cv::Scalar c = trackColor(t.id);
                for (auto &p : PAIRS) {
                    const cv::Point3f &a = kp[p[0]], &b = kp[p[1]];
                    if (a.z > 0.3f && b.z > 0.3f)
                        cv::line(frame, cv::Point((int)a.x,(int)a.y),
                                 cv::Point((int)b.x,(int)b.y), c, 2, cv::LINE_AA);
                }
                for (const auto &p : kp)
                    if (p.z > 0.3f)
                        cv::circle(frame, cv::Point((int)p.x,(int)p.y), 3, c, -1);
            }
        }
        drawOverlay(frame, inferMs, fpsNow);

        // fight overlay: red banner (kept until next clip decision)
        if (fightState_)
            cv::putText(frame, cv::format("FIGHT %.2f", fightScore_),
                        cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // fall overlay: red box + label on sticky tracks
        for (const auto &t : tracks) {
            if (!fallLife_.count(t.id)) continue;
            cv::Rect r((int)t.tlwh.x, (int)t.tlwh.y,
                       (int)t.tlwh.width, (int)t.tlwh.height);
            cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 3);
            cv::putText(frame, "FALL", cv::Point(r.x, std::max(12, r.y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        // deep copy + UI emit rate limit: monitors display at 12 fps
        // (industrial standard); keeps GUI load flat regardless of channel
        // count or inference fps, and skips the clone for dropped frames
        double nowMs = std::chrono::duration<double, std::milli>(
                           t1 - tStart).count();
        if (nowMs - lastUiMs_ < 80) continue;   // ~12.5 fps UI
        lastUiMs_ = nowMs;
        emit frameReady(frame.clone(), fpsNow, frameCount_, inferMs, (int)tracks.size());
    }

    cap_.release();
    capThread_.join();
    auto tEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    double avgFps = totalMs > 0 ? frameCount_ * 1000.0 / totalMs : 0.0;
    std::cout << "[VideoWorker] done: " << frameCount_ << " frames in " << totalMs
              << " ms | " << avgFps << " FPS\n";
    emit finished();
}

void VideoWorker::capLoop() {
    // network sources (rtsp:// and web-page cameras) use the process-isolated
    // ffmpeg pipe decoder; local video files keep the direct OpenCV path
    bool network = cfg_.video_path.rfind("rtsp://", 0) == 0 ||
                   cfg_.video_path.rfind("http://", 0) == 0 ||
                   cfg_.video_path.rfind("https://", 0) == 0;
    if (network) { pipeLoop(); return; }
    bool realtime = isCameraIndex(cfg_.video_path);
    const size_t qMax = realtime ? 2 : 4;
    bool announced = false;   // srcOpened_ reported to run() exactly once

    // Industrial NVR behaviour: live sources (camera index / rtsp://) NEVER
    // stop on network blips — reconnect with a 2 s interval until stopped.
    // File sources end at EOF (backlog analysis).
    while (!stopFlag_) {
        bool ok;
        if (realtime) {
            ok = cap_.open(atoi(cfg_.video_path.c_str()));
            if (ok) applyMaxCameraResolution(cap_);   // 摄像头默认跑最大分辨率
        } else {
            ok = cap_.open(cfg_.video_path, cv::CAP_FFMPEG);
        }
        if (!ok) {
            if (!realtime) {
                std::lock_guard<std::mutex> lk(qMtx_);
                srcErr_ = "Cannot open source: " + cfg_.video_path;
                srcOpened_ = true;
                qCv_.notify_all();
                return;
            }
            if (!announced) {   // let run() start; frames follow on reconnect
                std::lock_guard<std::mutex> lk(qMtx_);
                srcFps_ = 25.0;
                srcOpened_ = true;
                announced = true;
                qCv_.notify_all();
            }
            fprintf(stderr, "[source] connect failed, retry in 2 s: %s\n",
                    cfg_.video_path.c_str());
            fflush(stderr);
            emit sourceError("视频源连接失败，2 秒后自动重连");
            for (int i = 0; i < 20 && !stopFlag_; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (realtime && announced)
            fprintf(stderr, "[source] (re)connected\n"), fflush(stderr);

        if (!announced) {
            double fps = cap_.get(cv::CAP_PROP_FPS);
            if (fps <= 1e-3 || fps > 240) fps = 25.0;
            std::lock_guard<std::mutex> lk(qMtx_);
            srcFps_ = fps;
            srcW_ = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
            srcH_ = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
            srcOpened_ = true;
            announced = true;
            qCv_.notify_all();
        }

        // read loop: realtime drops stale frames (bounded latency); file uses
        // bounded backpressure (no frame skipped during analysis)
        auto openedAt = std::chrono::steady_clock::now();
        while (!stopFlag_) {
            // OpenCV FFMPEG backend degrades on long-running network streams
            // (observed segfault after ~11 min inside AVIWriteContainer on a
            // live rtsp source) — rebuild the decoder session every 8 minutes
            // before it breaks (NVR-style periodic decoder reconnect; the
            // outer loop reopens invisibly, analysis just skips a moment)
            if (realtime && std::chrono::steady_clock::now() - openedAt >
                    std::chrono::minutes(8)) {
                cap_.release();
                break;
            }
            cv::Mat frame;
            cap_ >> frame;
            if (frame.empty()) break;   // EOF (file) / stream drop (live)
            {
                std::unique_lock<std::mutex> lk(qMtx_);
                if (realtime) {
                    while (q_.size() >= qMax) q_.pop_front();
                } else {
                    qCv_.wait(lk, [this, qMax] { return q_.size() < qMax || stopFlag_; });
                    if (stopFlag_) break;
                }
                q_.push_back(frame);
            }
            qCv_.notify_one();
        }
        if (!realtime) break;           // file fully analysed -> finish

        fprintf(stderr, "[source] stream dropped, reconnecting in 2 s\n");
        fflush(stderr);
        emit sourceError("视频流断开，自动重连中");
        cap_.release();
        for (int i = 0; i < 20 && !stopFlag_; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cap_.release();
    {
        std::lock_guard<std::mutex> lk(qMtx_);
        srcEnded_ = true;
    }
    qCv_.notify_all();
}

// process-isolated decode for rtsp://: an ffmpeg child process pulls the
// network stream and emits raw BGR frames on stdout; this thread reads them.
// - ffmpeg crash / stream drop -> fread short read -> child restarts (NVR
//   reconnect loop); the app itself can never segfault inside a decoder
// - ffmpeg -t 600 -> the child exits after 10 min by itself -> fresh decoder
//   session every time (no long-run codec state degradation)
// - fixed 1280x720 output -> pipe frames split by fixed byte count
void VideoWorker::pipeLoop() {
    const int W = 1280, H = 720;
    const size_t frameBytes = (size_t)W * H * 3;
    const size_t qMax = 2;
    std::vector<uint8_t> buf(frameBytes);
    bool announced = false;
    int tokenFails = 0;   // 连续 token 解析失败次数 -> 指数退避

    while (!stopFlag_) {
        // 网页型公共摄像头：每次重建解码器时重新解析一次当前 token。
        // token 短命（实测几十秒即失效），绝不能写死，必须现取现用。
        // 关键：解析失败必须退避！固定 2 秒重试几百次会把源打成 429 彻底封禁，
        // 然后无限失败 -> 画面永久停住（实测事故）。指数退避到 30s，429 更长。
        std::string input = cfg_.video_path;
        bool isSkyline = input.rfind("https://www.skylinewebcams.com", 0) == 0;
        if (isSkyline) {
            int http = 0;
            std::string resolved = resolveSkylineStream(input, &http);
            if (resolved.empty()) {
                ++tokenFails;
                // 退避：2s -> 4 -> 8 -> 16 -> 30s 封顶；429 起步就 30s
                int waitSec = (http == 429) ? 30
                            : std::min(2 << std::min(tokenFails, 4), 30);
                // 日志降频：前 3 次 + 之后每 10 次才记一条，避免刷屏
                if (tokenFails <= 3 || tokenFails % 10 == 0) {
                    emit sourceError(http == 429
                        ? QString("摄像头限流(%1)，%2秒后重试").arg(http).arg(waitSec)
                        : QString("摄像头页面解析失败，%1秒后重试").arg(waitSec));
                }
                for (int i = 0; i < waitSec * 10 && !stopFlag_; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            tokenFails = 0;   // 成功拿到 token，重置退避
            input = resolved;
        }

        // -fflags 两次分开写：nobuffer(低延迟) 与 genpts+discardcorrupt(容错)
        // 属于不同作用域，必须分两次传；合并成一个会因作用域冲突导致某些源
        // (如纽约时代广场 ts 分片缺 SPS) 解码出 0 帧 -> 画面静止。
        // +genpts 重建时间戳，+discardcorrupt 跳过损坏分片，实测修复时代广场。
        std::string cmd =
            "ffmpeg -hide_banner -loglevel error "
            "-user_agent \"Mozilla/5.0\" "
            "-fflags nobuffer -fflags +genpts+discardcorrupt -an -i \"" + input + "\" "
            "-vf fps=15,scale=1280:720 -pix_fmt bgr24 -f rawvideo -t 600 -";
        FfmpegPipe pipe(cmd, this);
        if (!pipe.ok) {
            if (!announced) {
                std::lock_guard<std::mutex> lk(qMtx_);
                srcFps_ = 15.0; srcW_ = W; srcH_ = H;
                srcOpened_ = true; announced = true;
                qCv_.notify_all();
            }
            emit sourceError("ffmpeg 解码器启动失败，2 秒后重试");
            for (int i = 0; i < 20 && !stopFlag_; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (announced)
            fprintf(stderr, "[source] (re)connected\n"), fflush(stderr);
        if (!announced) {
            std::lock_guard<std::mutex> lk(qMtx_);
            srcFps_ = 15.0; srcW_ = W; srcH_ = H;
            srcOpened_ = true; announced = true;
            qCv_.notify_all();
        }

        // frame read loop: fixed-size raw frames off the pipe; realtime queue
        // semantics (drop stale frames for bounded latency).
        // readFrame 带 10 s 停滞看门狗：token 过期后 ffmpeg 卡在拉流重试、既不
        // 退出也不吐数据时，这里会超时退出 -> 析构杀掉卡死子进程 -> 重建会话
        // 重新解析 token -> 画面恢复动态（不再永久定格）。
        size_t gotAny = 0;
        bool stalled = false;
        while (!stopFlag_) {
            size_t got = pipe.readFrame(buf.data(), frameBytes, 10000);
            if (got < frameBytes) {
                // 区分真断流（子进程退出）和卡死（子进程还在、但零字节超时）
                stalled = !stopFlag_ && pipe.hProc &&
                          WaitForSingleObject(pipe.hProc, 0) == WAIT_TIMEOUT;
                break;
            }
            gotAny++;
            cv::Mat frame(H, W, CV_8UC3, buf.data());
            {
                std::unique_lock<std::mutex> lk(qMtx_);
                while (q_.size() >= qMax) q_.pop_front();
                if (stopFlag_) break;
                q_.push_back(frame.clone());   // buf is reused -> clone
            }
            qCv_.notify_one();
        }
        // FfmpegPipe 析构：关闭读端 -> 杀掉卡死/退出的 ffmpeg -> 无残留
        // - 停滞：立即重建并重新解析 token（画面曾流畅，只是流断）
        // - 一帧都没出（源不可达）：退避 2 s 再重试，避免刷屏
        if (!stopFlag_) {
            if (stalled) {
                emit sourceError("视频流卡顿，正在自动重连");
            } else if (gotAny == 0) {
                emit sourceError("视频源连接失败，2 秒后自动重连");
                for (int i = 0; i < 20 && !stopFlag_; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
}

void VideoWorker::drawTracks(cv::Mat &f, const std::vector<Trk> &tracks) {
    for (const auto &t : tracks) {
        cv::Scalar c = trackColor(t.id);
        cv::Rect r((int)t.tlwh.x, (int)t.tlwh.y,
                   (int)t.tlwh.width, (int)t.tlwh.height);
        cv::rectangle(f, r, c, 2);
        cv::putText(f, "ID:" + std::to_string(t.id),
                    cv::Point(r.x, std::max(12, r.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv::LINE_AA);
        // skeleton for this track is not tracked per-id here (phase 2 polish)
    }
}

void VideoWorker::drawOverlay(cv::Mat &f, double inferMs, double fps) {
    (void)fps;   // first line now shows inference time only
    cv::putText(f, cv::format("infer %.1f ms", inferMs),
                cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    // alert banner (latest alert, 5 s, red on dark strip)
    double nowMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::string alert;
    {
        std::lock_guard<std::mutex> lk(alertBannerMtx_);
        if (nowMs - latestAlertMs_ < 5000) alert = latestAlert_;
    }
    if (!alert.empty()) {
        int bl = cv::getTextSize(alert, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, nullptr).width;
        cv::rectangle(f, cv::Rect(0, 40, std::min(bl + 20, f.cols), 44),
                      cv::Scalar(30, 30, 30), -1);
        cv::putText(f, alert, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX,
                    0.9, cv::Scalar(60, 60, 255), 2, cv::LINE_AA);
    }
}
