#include "MainWindow.h"
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <QPixmap>
#include <QPushButton>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QImage>
#include <QPainter>
#include <QCoreApplication>
#include <QSettings>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMenuBar>
#include <QStatusBar>
#include <QGuiApplication>
#include <QScreen>
#include <QDialog>
#include <QFormLayout>
#include <QFileDialog>
#include <QInputDialog>
#include <QDate>
#include <QDateTime>
#include <QSpinBox>
#include <QTime>
#include <QDateTime>
#include <QHash>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/logger.hpp>

static QString modelFile(const char *name) {
    QString p = QDir(QCoreApplication::applicationDirPath()).filePath(name);
    return QFile::exists(p) ? p : QString();
}

// video label with fixed layout hints: keeps splitter/box from being
// pushed around by pixmap-sized minimumSizeHint.
// Also draws a centered spinning loading indicator while a channel waits
// for its first decoded frame (stream switch / reconnect / startup).
class VideoLabel : public QLabel {
public:
    explicit VideoLabel(QWidget *parent = nullptr) : QLabel(parent) {}
    QSize minimumSizeHint() const override { return QSize(320, 240); }
    QSize sizeHint() const override { return QSize(960, 540); }

    void setLoading(bool on) {
        if (loading_ == on) return;
        loading_ = on;
        angle_ = 0;
        update();
    }
    bool loading() const { return loading_; }
    void advance() {                       // called by the global spinner timer
        if (!loading_) return;
        angle_ = (angle_ + 30) % 360;
        update();
    }

protected:
    void paintEvent(QPaintEvent *e) override {
        QLabel::paintEvent(e);             // pixmap / text first
        if (!loading_) return;
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        // dim the cell so the spinner reads clearly over any stale frame
        p.fillRect(rect(), QColor(0, 0, 0, 110));
        int side = qMin(width(), height());
        int r = qMax(18, side / 12);       // ring radius scales with cell size
        QPoint c = rect().center();
        QRectF ring(c.x() - r, c.y() - r, 2 * r, 2 * r);
        QPen base(QColor(255, 255, 255, 70));
        base.setWidthF(qMax(2.0, side * 0.008));
        base.setCapStyle(Qt::RoundCap);
        p.setPen(base);
        p.drawArc(ring, 0, 5760);          // full faint ring (angle in 1/16 deg)
        QPen arc(QColor("#5fa8ff"));
        arc.setWidthF(qMax(2.0, side * 0.008));
        arc.setCapStyle(Qt::RoundCap);
        p.setPen(arc);
        p.drawArc(ring, angle_ * 16, 110 * 16);   // rotating bright arc
        QFont f = font();
        f.setPointSizeF(qMax(9.0, side * 0.030));
        p.setFont(f);
        p.setPen(QColor(230, 230, 230));
        p.drawText(QRectF(rect().x(), c.y() + r + 10, rect().width(), 40),
                   Qt::AlignHCenter | Qt::AlignTop, QStringLiteral("加载中..."));
    }

private:
    bool loading_ = false;
    int angle_ = 0;
};

// ---- generic small parameter dialog ----
struct ParamItem {
    const char *label;
    double min, max, step, value;
    bool isInt;
};

static bool paramDialog(QWidget *parent, const QString &title,
                        std::vector<ParamItem> &items) {
    QDialog dlg(parent);
    dlg.setWindowTitle(title);
    QFormLayout *form = new QFormLayout(&dlg);
    std::vector<QDoubleSpinBox *> dspins;
    std::vector<QSpinBox *> ispins;
    for (const auto &it : items) {
        if (it.isInt) {
            QSpinBox *s = new QSpinBox(&dlg);
            s->setRange((int)it.min, (int)it.max);
            s->setValue((int)it.value);
            form->addRow(it.label, s);
            ispins.push_back(s);
        } else {
            QDoubleSpinBox *s = new QDoubleSpinBox(&dlg);
            s->setRange(it.min, it.max);
            s->setSingleStep(it.step);
            s->setDecimals(2);
            s->setValue(it.value);
            form->addRow(it.label, s);
            dspins.push_back(s);
        }
    }
    QPushButton *ok = new QPushButton("确定", &dlg);
    QPushButton *cancel = new QPushButton("取消", &dlg);
    QHBoxLayout *hl = new QHBoxLayout;
    hl->addWidget(ok);
    hl->addWidget(cancel);
    form->addRow(hl);
    QObject::connect(ok, &QPushButton::clicked, &dlg, &QDialog::accept);
    QObject::connect(cancel, &QPushButton::clicked, &dlg, &QDialog::reject);
    if (dlg.exec() != QDialog::Accepted) return false;
    size_t di = 0, ii = 0;
    for (auto &it : items)
        it.value = it.isInt ? (double)ispins[ii++]->value() : dspins[di++]->value();
    return true;
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    loadSettings();      // restore saved behavior params before UI is built
    setupUI();
}

MainWindow::~MainWindow() {
    if (running_) {
        for (auto &ch : chans_) if (ch.worker) ch.worker->stop();
        for (auto &ch : chans_) if (ch.worker) ch.worker->wait(10000);
    }
}

void MainWindow::setupUI() {
    updateTitle();
    resize(1360, 780);

    // menu: 文件 -> 添加视频源(可多次,加载即分析) / 移除视频源 / 清空
    QMenu *mFile = menuBar()->addMenu("文件");
    QAction *actOpen = mFile->addAction("添加视频源");
    actOpen->setShortcut(QKeySequence::Open);
    connect(actOpen, &QAction::triggered, this, &MainWindow::onAddSource);
    QAction *actDel = mFile->addAction("移除视频源");
    connect(actDel, &QAction::triggered, this, &MainWindow::onRemoveSource);
    QAction *actClr = mFile->addAction("清空全部源");
    connect(actClr, &QAction::triggered, this, &MainWindow::onClearSources);
    menuActions_ = {actOpen, actDel, actClr};

    // menu: 设置 -> 检测与行为设置（模型开关/阈值/隔帧，运行中可热更）
    QMenu *mSet = menuBar()->addMenu("设置");
    QAction *actCfg = mSet->addAction("检测与行为设置");
    connect(actCfg, &QAction::triggered, this, &MainWindow::onSettings);
    actSet_ = actCfg;

    // menu: 连接 -> 真实摄像头流清单（点击切换，持续实时推理）。
    // 新接入的摄像头（场地/客户现场），往 cams[] 加一行即可
    // 公共场所人流监控（HLS 实时流，非视频文件）。
    // token 由 VideoWorker 每次开流前从页面动态解析，这里只存稳定页面地址。
    QMenu *mConn = menuBar()->addMenu("连接");
    connectMenu_ = mConn;
    // 本地摄像头只在启动时枚举一次（插在分隔符锚点之前，公共源在其后）。
    // 展开菜单不再重新探测 —— 每次点开都 open/read 十个索引会把菜单卡住；
    // 需要重新扫描时点菜单底部的"重新扫描本地摄像头"。
    QAction *localAnchor = mConn->addSeparator();
    localCamAnchor_ = localAnchor;
    QTimer::singleShot(300, this, &MainWindow::refreshLocalCameras);
    // 新接入的摄像头（场地/客户现场），往 cams[] 加一行即可
    static const struct { const char *place; const char *url; } cams[] = {
        {"意大利·威尼斯 圣马可广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/veneto/venezia/piazza-san-marco.html"},
        {"意大利·米兰 大教堂广场（Duomo 人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lombardia/milano/duomo-milano.html"},
        {"西班牙·马德里 太阳门广场（Puerta del Sol 人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/espana/comunidad-de-madrid/madrid/puerta-del-sol-tio-pepe.html"},
        {"意大利·罗马 威尼斯广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/piazza-venezia.html"},
        {"意大利·罗马 特莱维喷泉（景点人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/fontana-di-trevi.html"},
        {"意大利·罗马 鲜花广场（Campo de' Fiori 人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/campo-de-fiori.html"},
        {"意大利·那不勒斯 加里波第广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/campania/napoli/napoli-piazza-garibaldi.html"},
        {"意大利·的里雅斯特 意大利统一广场（公共广场 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/friuli-venezia-giulia/trieste/piazza-unita-italia.html"},
        {"意大利·佛罗伦萨 主教座堂广场（Duomo 人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/toscana/firenze/piazza-duomo-firenze.html"},
        {"美国·纽约 时代广场（Times Square 人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/united-states/new-york/new-york/times-square.html"},
        {"马耳他·瓦莱塔 圣乔治广场（公共广场 HLS）",
         "https://www.skylinewebcams.com/en/webcam/malta/malta/valletta/st-george-square-valletta.html"},
        {"以色列·耶路撒冷 西墙广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/israel/jerusalem-district/jerusalem/western-wall.html"},
        {"意大利·罗马 纳沃纳广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/piazza-navona.html"},
        {"希腊·雅典 宪法广场与 Ermou 商业街（市中心人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/ellada/atiki/athina/ermou-street-syntagma-square.html"},
        {"意大利·罗马 卡武尔广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/piazza-cavour.html"},
        {"意大利·罗马 复兴广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/piazza-risorgimento.html"},
        {"意大利·米兰 五日广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lombardia/milano/piazza-cinque-giornate.html"},
        {"意大利·米兰 斯福尔扎城堡广场（公共广场 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lombardia/milano/piazza-duca-dsosta.html"},
        {"意大利·米兰 圣巴比拉广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/lombardia/milano/piazza-san-babila.html"},
        {"意大利·那不勒斯 公民投票广场（公共广场 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/campania/napoli/piazza-del-plebiscito.html"},
        {"意大利·博洛尼亚 解放广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/italia/emilia-romagna/bologna/sasso-marconi-piazza-dei-martiri-della-liberazione.html"},
        {"西班牙·马德里 阿尔卡拉大街（Gran Vía 商业街人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/espana/comunidad-de-madrid/madrid/calle-alcala.html"},
        {"西班牙·马德里 卡亚奥广场（公共广场人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/espana/comunidad-de-madrid/madrid/madrid-plaza-del-callao.html"},
        {"西班牙·塞维利亚 圣弗朗西斯科广场（公共广场 HLS）",
         "https://www.skylinewebcams.com/en/webcam/espana/andalucia/sevilla/siviglia-plaza-san-francisco.html"},
        {"秘鲁·库斯科 武器广场（Plaza Mayor 人流 HLS）",
         "https://www.skylinewebcams.com/en/webcam/peru/cusco/cusco/plaza-mayor.html"},
        {"越南·义安省荣市 黄梅镇（沿河公路 HLS）",
         "https://www.skylinewebcams.com/en/webcam/vietnam/bac-trung-bo/nghe-an/hoang-mai-town-nghe-an.html"},
    };
    for (const auto &c : cams) {
        QAction *a = mConn->addAction(QString::fromUtf8(c.place));
        a->setData(QString::fromUtf8(c.url));
        a->setToolTip(QString::fromUtf8(c.url));
        connect(a, &QAction::triggered, this, &MainWindow::onConnectSource);
    }

    // 手动重扫（启动只枚举一次，插拔摄像头后点这里，不必重启软件）
    mConn->addSeparator();
    QAction *actRescan = mConn->addAction("重新扫描本地摄像头");
    connect(actRescan, &QAction::triggered, this, &MainWindow::onRescanCameras);

    // central: video grid only (alerts are drawn on the frames, logs go to files)
    QWidget *central = new QWidget(this);
    setCentralWidget(central);
    QHBoxLayout *mainLayout = new QHBoxLayout(central);
    gridHost_ = new QWidget(this);
    grid_ = new QGridLayout(gridHost_);
    grid_->setContentsMargins(0, 0, 0, 0);
    grid_->setSpacing(2);
    mainLayout->addWidget(gridHost_, 1);

    // status bar: run stats
    lblRun_ = new QLabel("通道: 0", this);
    statusBar()->addWidget(lblRun_);

    // debounce auto start: adding a source starts analysis automatically
    autoStart_ = new QTimer(this);
    autoStart_->setSingleShot(true);
    autoStart_->setInterval(150);   // short debounce: fast source switch (was 800ms)
    connect(autoStart_, &QTimer::timeout, this, [this] {
        if (!running_ && !sources_.isEmpty()) onStartStop();
    });

    // global spinner: drives the "loading" ring on every waiting cell.
    // One timer for all channels keeps GUI load flat at 16 routes.
    spinner_ = new QTimer(this);
    spinner_->setInterval(70);
    connect(spinner_, &QTimer::timeout, this, [this] {
        for (auto &ch : chans_)
            if (ch.view && ch.view->loading()) ch.view->advance();
    });
    spinner_->start();
}

void MainWindow::updateTitle() {
    setWindowTitle(placeName_.isEmpty()
                       ? QStringLiteral("行为监测")
                       : QStringLiteral("行为监测 - ") + placeName_);
}

void MainWindow::onAddSource() {
    QString path = QFileDialog::getOpenFileName(
        this, "选择视频", QStringLiteral("E:/DeepEpic/PPHuman-Qt/demo_input"),
        "Video (*.mp4 *.avi *.mkv *.mov);;All (*)");
    if (path.isEmpty()) return;
    placeName_.clear();          // local file: no "place", title = 行为监测
    updateTitle();
    replaceSource(path);         // replace current pipeline, no manual clear needed
}

void MainWindow::addSource(const QString &source) {
    if (source.isEmpty()) return;
    sources_ << source;
    rebuildGrid();
    if (running_) {
        // hot-add: this channel starts analysing immediately (NVR style)
        int i = sources_.size() - 1;
        if (!chans_[i].worker) {
            Config cfg = buildConfig(source);
            chans_[i].worker = makeChannelWorker(i);
            chans_[i].worker->init(cfg);
            chans_[i].worker->start();
            chans_[i].ended = false;
        }
        updateRunStats();
        writeLog("INFO", QString("热添加通道 %1: %2").arg(i + 1).arg(source));
    } else {
        autoStart_->start();   // load -> auto start (no start button)
    }
}

void MainWindow::replaceSource(const QString &source) {
    if (source.isEmpty()) return;
    // stop + join every running channel, then drop them all. stop() kills the
    // ffmpeg child synchronously so no orphan process / cmd window survives.
    for (auto &ch : chans_) {
        if (ch.worker) { ch.worker->stop(); ch.worker->wait(10000); ch.worker.reset(); }
    }
    running_ = false;
    sources_.clear();
    chans_.clear();
    addSource(source);           // debounced auto start -> single channel
}

void MainWindow::onRemoveSource() {
    if (sources_.isEmpty()) return;
    QStringList names;
    for (const QString &s : sources_) names << s;
    bool ok = false;
    QString pick = QInputDialog::getItem(this, "移除视频源",
                                         "选择要移除的源:", names, 0, false, &ok);
    if (!ok || pick.isEmpty()) return;
    // stop that channel's worker (hot remove, NVR style)
    for (int i = 0; i < sources_.size(); ++i) {
        if (sources_[i] != pick) continue;
        if (i < (int)chans_.size() && chans_[i].worker) {
            chans_[i].worker->stop();
            chans_[i].worker->wait(10000);
            chans_[i].worker.reset();
        }
        sources_.removeAt(i);
        break;
    }
    rebuildGrid();
    if (sources_.isEmpty() && running_) {
        running_ = false;   // all channels already stopped above
        for (auto *a : menuActions_) a->setEnabled(true);
        actSet_->setEnabled(true);
    }
    updateRunStats();
    writeLog("INFO", "移除通道: " + pick);
}

void MainWindow::onClearSources() {
    for (auto &ch : chans_) if (ch.worker) { ch.worker->stop(); ch.worker->wait(10000); }
    running_ = false;
    sources_.clear();
    chans_.clear();
    placeName_.clear();
    updateTitle();
    rebuildGrid();
    for (auto *a : menuActions_) a->setEnabled(true);
    actSet_->setEnabled(true);
    updateRunStats();
    writeLog("INFO", "已清空全部视频源");
}

// N sources -> automatic rows x cols (1:1x1, 2:2x1, 3-4:2x2, 5-6:3x2, 9:3x3)
void MainWindow::rebuildGrid() {
    // clear old cells (labels are owned by the layout)
    while (QLayoutItem *it = grid_->takeAt(0)) {
        if (QWidget *w = it->widget()) w->deleteLater();
        delete it;
    }
    // labels are rebuilt below; stale pointers MUST be cleared — they point
    // to deleteLater'd widgets until the event loop destroys them
    for (auto &ch : chans_) ch.view = nullptr;
    int n = sources_.size();
    if (n == 0) {
        QLabel *hint = new QLabel("未添加视频源\n（文件 → 添加视频源）", gridHost_);
        hint->setAlignment(Qt::AlignCenter);
        hint->setStyleSheet("color:#888;font-size:16pt;background:#1a1a1a;");
        grid_->addWidget(hint, 0, 0);
        chans_.clear();
        return;
    }
    int cols = (int)std::ceil(std::sqrt((double)n));
    int rows = (int)std::ceil((double)n / cols);
    chans_.resize(n);
    for (int i = 0; i < n; ++i) {
        if (!chans_[i].view) {
            auto *v = new VideoLabel(gridHost_);
            v->setScaledContents(true);
            v->setStyleSheet("background-color:#1a1a1a;border:1px solid #444;");
            v->setAlignment(Qt::AlignCenter);
            v->setText(QString("摄像头 %1").arg(i + 1));
            // waiting for first frame: show the spinning loading ring
            v->setLoading(true);
            chans_[i].view = v;
        }
        chans_[i].view->setParent(gridHost_);
        grid_->addWidget(chans_[i].view, i / cols, i % cols);
    }
}

void MainWindow::startAll() {
    onStartStop();
}

void MainWindow::setAnalyzeEvery(int n) {
    if (n >= 1) { cfg_.analyzeEvery = n; applyRuntimeToAll(); }
}

Config MainWindow::buildConfig(const QString &videoPath) const {
    Config cfg = cfg_;   // behavior switches / thresholds / frame skip / tuning
    // Windows ANSI codepage (GBK) so OpenCV can open Chinese paths
    cfg.video_path = videoPath.toLocal8Bit().toStdString();
    cfg.mot_engine_path   = modelFile("models/mot.engine").toStdString();
    cfg.kpt_engine_path   = modelFile("models/kpt.engine").toStdString();
    cfg.smoke_engine_path = modelFile("models/smoking.engine").toStdString();
    cfg.fight_engine_path = modelFile("models/ppTSM.engine").toStdString();
    cfg.fall_engine_path  = modelFile("models/STGCN.engine").toStdString();
    return cfg;
}

// hot-apply settings: thresholds/switches/interval -> all running workers
void MainWindow::applyRuntimeToAll() {
    Config cfg = buildConfig(QString());
    for (auto &ch : chans_) {
        if (!ch.worker) continue;
        ch.worker->setRuntimeParams(cfg);
        ch.worker->loadIfNeeded(cfg);
    }
}

void MainWindow::onParamDialog(int idx) {
    switch (idx) {
    case 0: {   // crowd
        std::vector<ParamItem> items = {
            {"聚集人数阈值", 2, 50, 1, (double)cfg_.crowdCount, true},
            {"持续时长(秒)", 1, 60, 0.5, cfg_.crowdSec, false},
        };
        if (paramDialog(this, "人群聚集参数", items)) {
            cfg_.crowdCount = (int)items[0].value;
            cfg_.crowdSec   = items[1].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    case 1: {   // loiter
        std::vector<ParamItem> items = {
            {"停留时长(秒)", 2, 120, 1, cfg_.loiterSec, false},
            {"移动范围(px)", 10, 500, 5, cfg_.loiterMove, false},
        };
        if (paramDialog(this, "徘徊逗留参数", items)) {
            cfg_.loiterSec  = items[0].value;
            cfg_.loiterMove = items[1].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    case 2: {   // run
        std::vector<ParamItem> items = {
            {"速度阈值(px/s)", 50, 2000, 10, cfg_.runSpeed, false},
            {"最少帧数", 2, 30, 1, (double)cfg_.runMinFrames, true},
        };
        if (paramDialog(this, "奔跑参数", items)) {
            cfg_.runSpeed     = items[0].value;
            cfg_.runMinFrames = (int)items[1].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    case 3: {   // posture
        std::vector<ParamItem> items = {
            {"躯干倾角(度)", 10, 120, 1, cfg_.postureAngle, false},
            {"对称性阈值", 0.05, 0.95, 0.05, cfg_.postureSym, false},
        };
        if (paramDialog(this, "异常姿态参数", items)) {
            cfg_.postureAngle = items[0].value;
            cfg_.postureSym   = items[1].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    case 4: {   // smoke
        std::vector<ParamItem> items = {
            {"检测间隔(秒)", 0.1, 5, 0.1, cfg_.smokeInterval, false},
        };
        if (paramDialog(this, "吸烟检测参数", items)) {
            cfg_.smokeInterval = items[0].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    case 5: {   // fight
        std::vector<ParamItem> items = {
            {"采样间隔(帧)", 1, 30, 1, (double)cfg_.fightSampleFreq, true},
        };
        if (paramDialog(this, "打架识别参数", items)) {
            cfg_.fightSampleFreq = (int)items[0].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    case 6: {   // fall
        std::vector<ParamItem> items = {
            {"报警保持(帧)", 10, 300, 5, (double)cfg_.fallKeepFrames, true},
        };
        if (paramDialog(this, "摔倒检测参数", items)) {
            cfg_.fallKeepFrames = (int)items[0].value;
            applyRuntimeToAll();
            saveSettings();          // persist tuning change
        }
        break;
    }
    }
}

// create a worker with all per-channel signal routing (index captured by value)
std::unique_ptr<VideoWorker> MainWindow::makeChannelWorker(int idx) {
    auto w = std::make_unique<VideoWorker>(this);
    connect(w.get(), &VideoWorker::frameReady, this,
        [this, idx](const cv::Mat &f, double fps, int c, double ms, int d) {
            Q_UNUSED(fps); Q_UNUSED(c); Q_UNUSED(ms); Q_UNUSED(d);
            showFrameAt(idx, f);
        }, Qt::QueuedConnection);
    connect(w.get(), &VideoWorker::behaviorAlert, this,
        [this, idx](const QString &type, const QString &detail) {
            // alerts live on the frame banner (worker-side); file log for trace
            writeLog(type, QString("[摄像头%1] %2").arg(idx + 1).arg(detail));
        }, Qt::QueuedConnection);
    connect(w.get(), &VideoWorker::sourceError, this,
        &MainWindow::onSourceError, Qt::QueuedConnection);
    connect(w.get(), &VideoWorker::finished, this,
        [this, idx] {
        chans_[idx].ended = true;
        updateRunStats();
        bool all = true;
        for (auto &c : chans_) if (c.worker && !c.ended) { all = false; break; }
        if (all && running_) {
            running_ = false;
            writeLog("INFO", "全部分析结束");
        }
    }, Qt::QueuedConnection);
    return w;
}

void MainWindow::onStartStop() {
    if (!running_) {
        int n = sources_.size();
        if (n == 0) return;
        Config base = buildConfig(QString());   // shared behavior config
        for (int i = 0; i < n; ++i) {
            Channel &ch = chans_[i];
            if (!ch.worker) ch.worker = makeChannelWorker(i);
            Config cfg = base;
            cfg.video_path = sources_[i].toLocal8Bit().toStdString();
            ch.worker->init(cfg);
            ch.worker->start();
            ch.ended = false;
            if (ch.view) ch.view->setLoading(true);   // wait for first frame
        }
        running_ = true;
        QString on;
        if (base.beh_crowd) on += " CROWD";
        if (base.beh_loiter) on += " LOITER";
        if (base.beh_run) on += " RUN";
        if (base.beh_posture) on += " POSTURE";
        if (base.beh_smoke) on += " SMOKE";
        if (base.beh_fight) on += " FIGHT";
        if (base.beh_fall) on += " FALL";
        writeLog("INFO", QString("开始分析 %1 路 | 启用:%2").arg(n).arg(on));
        updateRunStats();
    } else {
        for (auto &ch : chans_) if (ch.worker) ch.worker->stop();
        writeLog("INFO", "正在停止...");
    }
}

void MainWindow::updateRunStats() {
    int run = 0, total = 0;
    for (auto &ch : chans_) {
        if (!ch.worker) continue;
        total++;
        if (!ch.ended) run++;
    }
    lblRun_->setText(QString("通道: %1/%2 运行中").arg(run).arg(total));
}

void MainWindow::onSourceError(const QString &msg) {
    writeLog("ERROR", msg);
}

// local file log: exe_dir/logs/log_YYYYMMDD.txt (alerts shown on frames)
void MainWindow::writeLog(const QString &type, const QString &detail) {
    QDir dir(QCoreApplication::applicationDirPath());
    dir.mkpath("logs");
    QString day = QDate::currentDate().toString("yyyyMMdd");
    QFile f(dir.filePath(QString("logs/log_%1.txt").arg(day)));
    if (!f.open(QIODevice::Append | QIODevice::Text)) return;
    QString stamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    f.write(QString("[%1] [%2] %3\n").arg(stamp, type, detail).toUtf8());
}

// ---- settings persistence (QSettings ini next to the exe) ----
// All persisted knobs in one table: load / save / reset share it, so adding
// a new parameter is a single line. Engine paths and video sources are NOT
// persisted (they are machine/deployment specific).
namespace {
struct SettingEntry {
    const char *key;
    bool  Config::*flag;      // behavior switch (null for numeric)
    float Config::*flt;       // float field (null for bool/int)
    int   Config::*num;       // int field (null for bool/float)
    bool  defFlag; float defFlt; int defInt;
};
// flag entries
static const SettingEntry kFlags[] = {
    {"beh_crowd",   &Config::beh_crowd,   nullptr, nullptr, true,  0, 0},
    {"beh_loiter",  &Config::beh_loiter,  nullptr, nullptr, true,  0, 0},
    {"beh_run",     &Config::beh_run,     nullptr, nullptr, true,  0, 0},
    {"beh_posture", &Config::beh_posture, nullptr, nullptr, true,  0, 0},
    {"beh_smoke",   &Config::beh_smoke,   nullptr, nullptr, true,  0, 0},
    {"beh_fight",   &Config::beh_fight,   nullptr, nullptr, true,  0, 0},
    {"beh_fall",    &Config::beh_fall,    nullptr, nullptr, true,  0, 0},
};
static const SettingEntry kFloats[] = {
    {"th_crowd",   nullptr, &Config::th_crowd,   nullptr, false, 0.50f, 0},
    {"th_loiter",  nullptr, &Config::th_loiter,  nullptr, false, 0.50f, 0},
    {"th_run",     nullptr, &Config::th_run,     nullptr, false, 0.50f, 0},
    {"th_posture", nullptr, &Config::th_posture, nullptr, false, 0.50f, 0},
    {"th_smoke",   nullptr, &Config::th_smoke,   nullptr, false, 0.65f, 0},
    {"th_fight",   nullptr, &Config::th_fight,   nullptr, false, 0.50f, 0},
    {"th_fall",    nullptr, &Config::th_fall,    nullptr, false, 0.85f, 0},
    {"crowdSec",    nullptr, &Config::crowdSec,   nullptr, false, 3.0f,  0},
    {"loiterSec",   nullptr, &Config::loiterSec,  nullptr, false, 10.0f, 0},
    {"loiterMove",  nullptr, &Config::loiterMove, nullptr, false, 50.0f, 0},
    {"runSpeed",    nullptr, &Config::runSpeed,   nullptr, false, 400.0f,0},
    {"postureAngle",nullptr, &Config::postureAngle,nullptr, false, 60.0f, 0},
    {"postureSym",  nullptr, &Config::postureSym, nullptr, false, 0.30f, 0},
    {"smokeInterval",nullptr,&Config::smokeInterval,nullptr,false, 0.5f,  0},
};
static const SettingEntry kInts[] = {
    {"crowdCount",      nullptr, nullptr, &Config::crowdCount,      false, 0, 5},
    {"runMinFrames",    nullptr, nullptr, &Config::runMinFrames,    false, 0, 5},
    {"fightSampleFreq", nullptr, nullptr, &Config::fightSampleFreq, false, 0, 7},
    {"fallKeepFrames",  nullptr, nullptr, &Config::fallKeepFrames,  false, 0, 80},
    {"analyzeEvery",    nullptr, nullptr, &Config::analyzeEvery,    false, 0, 1},
    {"minPersonH",      nullptr, nullptr, &Config::minPersonH,      false, 0, 80},
};
QString settingsPath() {
    return QDir(QCoreApplication::applicationDirPath()).filePath("settings.ini");
}
} // namespace

void MainWindow::loadSettings() {
    QSettings s(settingsPath(), QSettings::IniFormat);
    for (const auto &e : kFlags)
        if (s.contains(e.key)) cfg_.*(e.flag) = s.value(e.key).toBool();
    for (const auto &e : kFloats)
        if (s.contains(e.key)) cfg_.*(e.flt) = s.value(e.key).toFloat();
    for (const auto &e : kInts)
        if (s.contains(e.key)) cfg_.*(e.num) = s.value(e.key).toInt();
}

void MainWindow::saveSettings() {
    QSettings s(settingsPath(), QSettings::IniFormat);
    for (const auto &e : kFlags) s.setValue(e.key, cfg_.*(e.flag));
    for (const auto &e : kFloats) s.setValue(e.key, cfg_.*(e.flt));
    for (const auto &e : kInts)   s.setValue(e.key, cfg_.*(e.num));
    s.sync();
}

void MainWindow::resetSettings() {
    for (const auto &e : kFlags) cfg_.*(e.flag) = e.defFlag;
    for (const auto &e : kFloats) cfg_.*(e.flt) = e.defFlt;
    for (const auto &e : kInts)   cfg_.*(e.num) = e.defInt;
    saveSettings();              // persist the defaults too
    applyRuntimeToAll();
}

void MainWindow::showFrameAt(int idx, const cv::Mat &frame) {
    if (idx < 0 || idx >= (int)chans_.size() || !chans_[idx].view) return;
    // UI render rate limit, adaptive to channel count (industrial NVR
    // displays: 15 fps for a few channels, 10 fps for dense grids —
    // keeps GUI load flat at 10-16 channels)
    static QHash<int, qint64> lastRender;
    qint64 interval = chans_.size() > 8 ? 100 : 66;
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - lastRender.value(idx, 0) < interval) return;
    lastRender.insert(idx, now);
    // dense grid: downscale before color-convert (5+ cells never show more
    // than ~1/4 of full HD on screen; converting 16x1080p/s would swamp the
    // GUI thread — industrial NVRs render cell-sized buffers)
    cv::Mat small;
    const cv::Mat *src = &frame;
    if (chans_.size() > 4 && frame.cols > 960) {
        cv::resize(frame, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        src = &small;
    }
    cv::Mat rgb;
    cv::cvtColor(*src, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    chans_[idx].view->setLoading(false);   // first frame arrived -> stop spinner
    chans_[idx].view->setPixmap(QPixmap::fromImage(img));   // scaledContents fills area
}

// settings dialog: behavior switches + thresholds + frame skip (hot apply)
void MainWindow::onSettings() {
    QDialog dlg(this);
    dlg.setWindowTitle("检测与行为设置");
    dlg.setMinimumWidth(420);
    auto *lay = new QVBoxLayout(&dlg);

    struct Row { const char *name; };
    static const Row rows[7] = {
        {"人群聚集 (CROWD)"}, {"徘徊逗留 (LOITER)"}, {"奔跑 (RUN)"},
        {"异常姿态 (POSTURE)"}, {"吸烟 (SMOKE)"}, {"打架 (FIGHT)"},
        {"摔倒 (FALL)"},
    };
    QCheckBox *chk[7];
    QDoubleSpinBox *conf[7];
    for (int i = 0; i < 7; ++i) {
        auto *row = new QWidget(&dlg);
        auto *hl = new QHBoxLayout(row);
        hl->setContentsMargins(0, 0, 0, 0);
        hl->setSpacing(4);
        chk[i] = new QCheckBox(rows[i].name, row);
        conf[i] = new QDoubleSpinBox(row);
        conf[i]->setRange(0.05, 0.95);
        conf[i]->setSingleStep(0.05);
        conf[i]->setFixedWidth(78);
        conf[i]->setToolTip("该行为参与的最低检测置信度（官方 crop_thresh 0.5）");
        // read current values from base config
        bool en; float th;
        switch (i) {
        case 0: en = cfg_.beh_crowd;   th = cfg_.th_crowd;   break;
        case 1: en = cfg_.beh_loiter;  th = cfg_.th_loiter;  break;
        case 2: en = cfg_.beh_run;     th = cfg_.th_run;     break;
        case 3: en = cfg_.beh_posture; th = cfg_.th_posture; break;
        case 4: en = cfg_.beh_smoke;   th = cfg_.th_smoke;   break;
        case 5: en = cfg_.beh_fight;   th = cfg_.th_fight;   break;
        case 6: en = cfg_.beh_fall;    th = cfg_.th_fall;    break;
        }
        chk[i]->setChecked(en);
        conf[i]->setValue(th);
        QPushButton *btnCfg = new QPushButton("参数", row);
        btnCfg->setFixedWidth(56);
        hl->addWidget(chk[i], 1);
        hl->addWidget(conf[i]);
        hl->addWidget(btnCfg);
        lay->addWidget(row);
        connect(btnCfg, &QPushButton::clicked, this, [this, i] { onParamDialog(i); });
    }

    // frame skip (multi-channel deployment, industrial standard)
    auto *everyRow = new QWidget(&dlg);
    auto *hlE = new QHBoxLayout(everyRow);
    hlE->setContentsMargins(0, 0, 0, 0);
    QLabel *lblEvery = new QLabel("隔帧分析", everyRow);
    auto *spinEvery = new QSpinBox(everyRow);
    spinEvery->setRange(1, 10);
    spinEvery->setValue(cfg_.analyzeEvery);
    spinEvery->setToolTip("每 N 帧分析 1 帧（分析帧率=视频帧率/N）。\n"
                          "多路部署用 2（15fps 摄像头 -> 7.5fps 分析率），\n"
                          "单路实时分析保持 1");
    hlE->addWidget(lblEvery, 1);
    hlE->addWidget(spinEvery);
    lay->addWidget(everyRow);

    // far-scene guard: minimum person height for FALL/SMOKE/FIGHT
    auto *minRow = new QWidget(&dlg);
    auto *hlM = new QHBoxLayout(minRow);
    hlM->setContentsMargins(0, 0, 0, 0);
    QLabel *lblMin = new QLabel("最小目标高度(px)", minRow);
    auto *spinMin = new QSpinBox(minRow);
    spinMin->setRange(0, 400);
    spinMin->setValue(cfg_.minPersonH);
    spinMin->setToolTip("FALL/SMOKE/FIGHT 只分析不低于此高度的行人（防远景误报）。\n"
                        "远景广场建议 80~120；近景可设 0（不限制）。\n"
                        "0 = 关闭过滤，所有人参与行为分类");
    hlM->addWidget(lblMin, 1);
    hlM->addWidget(spinMin);
    lay->addWidget(minRow);

    auto *btns = new QHBoxLayout;
    QPushButton *restore = new QPushButton("恢复默认", &dlg);
    QPushButton *ok = new QPushButton("应用", &dlg);
    btns->addWidget(restore);
    btns->addWidget(ok, 1);
    lay->addLayout(btns);
    // restore: 立即写回默认到 cfg_ + 存盘 + 热更新；同时同步对话框控件
    connect(restore, &QPushButton::clicked, this, [this, &dlg, chk, conf, spinEvery, spinMin] {
        resetSettings();
        const bool defEn[7] = {true,true,true,true,true,true,true};
        const float defTh[7] = {0.50f,0.50f,0.50f,0.50f,0.65f,0.50f,0.50f};
        for (int i = 0; i < 7; ++i) { chk[i]->setChecked(defEn[i]); conf[i]->setValue(defTh[i]); }
        spinEvery->setValue(1);
        spinMin->setValue(80);
        writeLog("INFO", "参数已恢复默认并保存");
    });
    connect(ok, &QPushButton::clicked, &dlg, &QDialog::accept);
    if (dlg.exec() != QDialog::Accepted) return;

    // write back to base config + hot apply to all running channels
    for (int i = 0; i < 7; ++i) {
        bool en = chk[i]->isChecked();
        float th = (float)conf[i]->value();
        switch (i) {
        case 0: cfg_.beh_crowd = en;   cfg_.th_crowd = th;   break;
        case 1: cfg_.beh_loiter = en;  cfg_.th_loiter = th;  break;
        case 2: cfg_.beh_run = en;     cfg_.th_run = th;     break;
        case 3: cfg_.beh_posture = en; cfg_.th_posture = th; break;
        case 4: cfg_.beh_smoke = en;   cfg_.th_smoke = th;   break;
        case 5: cfg_.beh_fight = en;   cfg_.th_fight = th;   break;
        case 6: cfg_.beh_fall = en;    cfg_.th_fall = th;    break;
        }
    }
    cfg_.analyzeEvery = spinEvery->value();
    cfg_.minPersonH   = spinMin->value();
    applyRuntimeToAll();
    saveSettings();                          // persist to settings.ini
    writeLog("INFO", "设置已更新并保存（运行中即时生效）");
}

// menu 连接: 枚举本地摄像头（USB / 笔记本自带）——启动时一次，或用户点"重新扫描"。
// 不再挂在 aboutToShow 上：每展开一次菜单就要 open/read 十个索引，菜单会明显卡顿。
// 用 OpenCV 直接试开索引 i 探测（和 VideoWorker 打开摄像头同一后端，保证
// 菜单里列出的就一定能打开）。注意 Qt 的 QCameraInfo 在本机实测返回 0
// （DirectShow 枚举不到 MediaFoundation-only 的 UVC 摄像头），所以不用它。
// 探测成功即 release，不占用设备；探测失败的 index 跳过（编号不连续也正确）。
// 动态项插在分隔符锚点之前，永远排在网络摄像头上面；data=索引字符串，
// onConnectSource -> replaceSource -> capLoop 走 cap.open(i) 采集卡路径。
void MainWindow::refreshLocalCameras() {
    if (!connectMenu_ || !localCamAnchor_) return;
    // 清掉上一轮的本地项（不动分隔符之后的固定公共源）
    for (QAction *a : localCamActions_) {
        connectMenu_->removeAction(a);
        delete a;
    }
    localCamActions_.clear();

    // 临时静音 OpenCV 的 "Camera index out of range" 报错，探测完恢复
    auto oldLog = cv::utils::logging::getLogLevel();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    QVector<int> validIdx;
    for (int i = 0; i < 10; ++i) {           // 最多探测 10 个索引，够覆盖本地摄像头
        cv::VideoCapture cap;
        bool ok = false;
        try { ok = cap.open(i); } catch (...) { ok = false; }
        bool usable = false;
        if (ok) {
            cv::Mat f;
            usable = cap.read(f) && !f.empty();   // 真能出帧才算
            cap.release();
        }
        if (usable) validIdx << i;
    }
    cv::utils::logging::setLogLevel(oldLog);

    static const char *cn[] = {"一","二","三","四","五","六","七","八","九","十"};
    // validIdx 里 i 已升序；倒序插到锚点前，菜单里保持 一、二、三 自上而下
    for (int k = validIdx.size() - 1; k >= 0; --k) {
        int idx = validIdx[k];
        QString label = (k < 10)
            ? QString("本地摄像头%1").arg(QString::fromUtf8(cn[k]))
            : QString("本地摄像头%1").arg(k + 1);
        QAction *a = new QAction(label, connectMenu_);
        a->setData(QString::number(idx));    // 真实索引 -> cap.open(idx)
        a->setToolTip(QString("系统摄像头索引 %1").arg(idx));
        connect(a, &QAction::triggered, this, &MainWindow::onConnectSource);
        connectMenu_->insertAction(localCamAnchor_, a);
        localCamActions_.prepend(a);
    }
    if (validIdx.isEmpty()) {
        QAction *none = new QAction("（未检测到本地摄像头）", connectMenu_);
        none->setEnabled(false);
        connectMenu_->insertAction(localCamAnchor_, none);
        localCamActions_.prepend(none);
    }
    writeLog("INFO", QString("本地摄像头扫描完成: %1 个").arg(validIdx.size()));
}

// menu 连接: 手动重扫本地摄像头（插拔设备后不必重启软件）
void MainWindow::onRescanCameras() {
    refreshLocalCameras();
}

// menu 连接: switch the whole pipeline to the picked public camera
void MainWindow::onConnectSource() {
    QAction *act = qobject_cast<QAction*>(sender());
    if (!act) return;
    QString url = act->data().toString();
    QString place = act->text();
    // title shows place only, strip the parenthesized HLS/scene note.
    // 用字符串字面量（全角括号是多字节，字符字面量会被截断匹配失败）
    int cut = place.indexOf(QStringLiteral("（"));
    if (cut < 0) cut = place.indexOf('(');
    if (cut > 0) place = place.left(cut).trimmed();
    placeName_ = place;
    updateTitle();
    replaceSource(url);
    writeLog("INFO", "切换摄像头: " + place + "  " + url);
}
