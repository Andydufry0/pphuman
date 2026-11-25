#pragma once
#include <opencv2/core.hpp>
#include <vector>

// Shared data types for all model wrappers.
struct DetResult {
    int cls = 0;
    float score = 0.f;
    cv::Rect2f box;   // original frame coords
};

struct KeyPoints {
    std::vector<cv::Point3f> pts;   // COCO17: x, y, conf
};
