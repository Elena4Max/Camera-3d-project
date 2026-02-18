#pragma once
#include <opencv2/opencv.hpp>

void draw3DBox(
    cv::Mat& frame,
    const cv::Mat& K,
    const cv::Mat& dist,
    const cv::Mat& R,
    const cv::Mat& t,
    float X,
    float Y,
    float w,
    float l,
    float h);
