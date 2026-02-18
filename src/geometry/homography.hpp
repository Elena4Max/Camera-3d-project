#pragma once
#include <opencv2/opencv.hpp>

struct GroundModel
{
    cv::Mat H;
    cv::Mat Hinv;
};

GroundModel computeHomography(
    const std::vector<cv::Point2f>& groundPts,
    const std::vector<cv::Point2f>& imagePts);

cv::Point2f imageToGround(
    const cv::Point2f& imgPt,
    const GroundModel& model);

void extractPoseFromHomography(
    const cv::Mat& H,
    const cv::Mat& K,
    cv::Mat& R,
    cv::Mat& t);
