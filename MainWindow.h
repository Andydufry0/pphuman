#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QAction>
#include <QMenu>
#include <QGridLayout>
#include <QListWidget>
#include <QTimer>
#include <memory>
#include <vector>
#include "Config.h"
#include "VideoWorker.h"

class VideoLabel;   // defined in MainWindow.cpp; grid cell with loading ring

// PP-Human 行为监测主窗口（工业 NVR 风格）：
// 全屏视频网格（按源数自动排列）+ 顶部菜单（文件/设置）
// - 加载视频源即自动开始分析（无开始按钮）
// - 行为开关/阈值/隔帧集中在菜单"设置"对话框，支持运行中热更
// - 告警直接显示在监控画面上（横幅+标注）；日志写入本地 logs/ 目录
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;
    void addSource(const QString &source);   // append (command-line multi-source)
    void replaceSource(const QString &source);  // stop all + run only this source
    void startAll();                         // 启动全部通道（命令行自动调用）

private slots:
    void onAddSource();      // 菜单：添加视频源（可多次）
    void onRemoveSource();   // 移除选中源（运行中支持热移除）
    void onClearSources();   // 清空全部源
    void onStartStop();      // 启动/停止全部通道（自动开始内部调用）
    void onSettings();       // 菜单：检测与行为设置对话框
    void onConnectSource();  // 菜单：连接 -> 切换到该公共摄像头
    void refreshLocalCameras();  // 枚举本地摄像头并重建菜单项（启动时一次 / 手动重扫）
    void onRescanCameras();      // 菜单：强制重新扫描本地摄像头
    void onSourceError(const QString &msg);

private:
    std::unique_ptr<VideoWorker> makeChannelWorker(int idx);
    void setupUI();
    void onParamDialog(int idx);             // per-model settings dialog
    void setAnalyzeEvery(int n);             // frame skip (runtime)
    void rebuildGrid();                      // N 源 -> 自动行列网格
    void showFrameAt(int idx, const cv::Mat &frame);
    void writeLog(const QString &type, const QString &detail);
    Config buildConfig(const QString &videoPath) const;
    void applyRuntimeToAll();                // hot params -> all workers
    void updateTitle();                      // title bar: 行为监测 - <地点名>
    void loadSettings();                     // QSettings ini -> cfg_ (startup)
    void saveSettings();                     // cfg_ -> QSettings ini (on apply)
    void resetSettings();                    // restore official defaults
    void updateRunStats();                   // statusbar: 运行 x/y 路

    // base config: behavior switches / thresholds / frame skip / tuning.
    // video_path is filled per channel at start
    Config cfg_;

    // one channel per source: its own worker thread + grid cell
    struct Channel {
        VideoLabel *view = nullptr;
        std::unique_ptr<VideoWorker> worker;
        bool ended = true;                   // worker finished (file EOF / error)
    };
    std::vector<Channel> chans_;
    QStringList sources_;                    // video sources in add order
    QWidget *gridHost_ = nullptr;
    QGridLayout *grid_ = nullptr;
    QLabel *lblRun_ = nullptr;               // statusbar: 运行 x/y 路
    QTimer *autoStart_ = nullptr;            // debounce for auto start
    QTimer *spinner_ = nullptr;              // drives loading rings on waiting cells
    QList<QAction*> menuActions_;            // file menu (disabled while running)
    QAction *actSet_ = nullptr;              // settings menu action
    QMenu *connectMenu_ = nullptr;           // 连接菜单：本地摄像头动态段 + 固定公共源
    QAction *localCamAnchor_ = nullptr;      // 分隔符锚点：本地摄像头项插在它之前
    QList<QAction*> localCamActions_;        // 枚举出的本地摄像头项（启动 / 重扫时重建）
    bool running_ = false;
    QString placeName_;                      // current camera place -> window title
};
