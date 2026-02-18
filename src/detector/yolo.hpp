#pragma once
#include <opencv2/opencv.hpp>

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class YOLODetector
{
public:
    YOLODetector(const std::string& modelPath,
                 float confThreshold = 0.4f,
                 float nmsThreshold = 0.5f);

    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;
    float confThreshold;
    float nmsThreshold;
    int inputWidth = 640;
    int inputHeight = 640;
};
