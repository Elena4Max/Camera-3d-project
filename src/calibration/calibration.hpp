#pragma once
#include <opencv2/opencv.hpp>

struct CameraParameters
{
    cv::Mat K;
    cv::Mat dist;
    double reprojectionError;
};

CameraParameters runCalibration(
    const std::vector<std::string>& imagePaths,
    cv::Size boardSize,
    float squareSize);

void saveCamera(
    const std::string& filename,
    const CameraParameters& params);

void loadCamera(
    const std::string& filename,
    cv::Mat& K,
    cv::Mat& dist);
