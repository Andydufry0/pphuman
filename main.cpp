#include "MainWindow.h"
#include "crash_handler.h"
#include <QApplication>
#include <QTimer>
#include <QFile>
#include <QStringList>
#include <opencv2/core.hpp>

int main(int argc, char *argv[]) {
    installCrashHandler();   // crash evidence: symbolized stack trace to stderr
    QApplication app(argc, argv);
    // industrial dark theme: charcoal panels, light text (monitoring-room look)
    app.setStyleSheet(
        "QMainWindow, QDialog, QWidget { background-color: #232323; color: #dddddd; }"
        "QMenuBar { background-color: #1a1a1a; color: #dddddd; }"
        "QMenuBar::item { padding: 4px 10px; background: transparent; }"
        "QMenuBar::item:selected { background-color: #3a3a3a; }"
        "QMenu { background-color: #2a2a2a; color: #dddddd; border: 1px solid #4a4a4a; }"
        "QMenu::item { padding: 4px 24px; }"
        "QMenu::item:selected { background-color: #3d5068; }"
        "QMenu::separator { height: 1px; background: #444444; margin: 3px 6px; }"
        "QStatusBar { background-color: #1a1a1a; color: #aaaaaa; }"
        "QPushButton { background-color: #333333; color: #dddddd;"
        "  border: 1px solid #555555; border-radius: 3px; padding: 4px 12px; }"
        "QPushButton:hover { background-color: #3d3d3d; border-color: #6a6a6a; }"
        "QPushButton:pressed { background-color: #2a2a2a; }"
        "QCheckBox { color: #dddddd; spacing: 6px; }"
        "QDoubleSpinBox, QSpinBox { background-color: #2a2a2a; color: #dddddd;"
        "  border: 1px solid #555555; border-radius: 3px; padding: 2px 4px; }"
        "QListWidget { background-color: #2a2a2a; color: #dddddd;"
        "  border: 1px solid #555555; }"
        "QLabel { color: #dddddd; }"
        "QToolTip { background-color: #2a2a2a; color: #dddddd;"
        "  border: 1px solid #555555; padding: 3px; }");
    qRegisterMetaType<cv::Mat>("cv::Mat");   // queued frameReady across threads
    MainWindow w;
    w.showMaximized();   // always fits the screen (DPI-safe)

    // Command-line: <source1> [source2 ...]  (files / camera index / rtsp urls)
    // grid auto-arranged by count; frame-skip set in the UI before start
    QStringList args = app.arguments();
    for (int i = 1; i < args.size(); ++i) w.addSource(args.at(i));
    if (args.size() > 1) w.startAll();
    return app.exec();
}
