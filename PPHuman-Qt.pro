QT += core gui widgets
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PPHuman-Qt
TEMPLATE = app
CONFIG += c++17 windows
CONFIG -= app_bundle
msvc:QMAKE_CXXFLAGS += /utf-8   # UTF-8 source & execution charset (中文 UI 字符串)

SOURCES += \
    main.cpp \
    MainWindow.cpp \
    VideoWorker.cpp \
    PpDetModel.cpp \
    KptModel.cpp \
    ActionModels.cpp \
    FightModel.cpp \
    FallModel.cpp \
    ByteTracker.cpp \
    BehaviorDetector.cpp \
    TrtEngine.cpp

HEADERS += \
    MainWindow.h \
    VideoWorker.h \
    PPHuman.h \
    ModelTypes.h \
    PpDetModel.h \
    KptModel.h \
    ActionModels.h \
    FightModel.h \
    FallModel.h \
    ByteTracker.h \
    BehaviorDetector.h \
    TrtEngine.h \
    Config.h

# Paths
OPENCV = D:/ThirdLibrary/opencv/build
TRT    = E:/Deep/TensorRT/TensorRT-10.16.1.11
CUDA   = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9

INCLUDEPATH += $$OPENCV/include
INCLUDEPATH += $$TRT/include
INCLUDEPATH += "$$CUDA/include"
LIBS += -L$$OPENCV/x64/vc16/lib -lopencv_world4120
LIBS += -L$$quote($$TRT/lib) -lnvinfer_10 -lnvinfer_plugin_10
LIBS += -L$$quote($$CUDA/lib/x64) -lcudart

win32:LIBS += -luser32 -lgdi32 -lcomdlg32 -ldbghelp -lwinhttp
