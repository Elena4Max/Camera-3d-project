#include "calibration.hpp"

CameraParameters runCalibration(
    const std::vector<std::string>& imagePaths,
    cv::Size boardSize,
    float squareSize)
{
    std::vector<std::vector<cv::Point3f>> objPts;
    std::vector<std::vector<cv::Point2f>> imgPts;

    std::vector<cv::Point3f> obj;

    for (int i=0;i<boardSize.height;i++)
        for (int j=0;j<boardSize.width;j++)
            obj.emplace_back(
                j*squareSize,
                i*squareSize,
                0);

    cv::Size imgSize;

    for (auto& path : imagePaths)
    {
        cv::Mat img =
            cv::imread(path);

        imgSize = img.size();

        std::vector<cv::Point2f> corners;
        cv::Mat gray;

        cv::cvtColor(img,gray,
                     cv::COLOR_BGR2GRAY);

        if (cv::findChessboardCornersSB(
                gray,boardSize,corners))
        {
            imgPts.push_back(corners);
            objPts.push_back(obj);
        }

        if (!cv::findChessboardCornersSB(gray, boardSize, corners))
        {
            std::cout << "Chessboard NOT found in: "
                    << path << std::endl;
            continue;
        }
        else
        {
            std::cout << "Chessboard found in: "
                    << path << std::endl;
        }

    }

    cv::Mat K,dist;
    std::vector<cv::Mat> r,t;

    double err =
        cv::calibrateCamera(
            objPts,imgPts,
            imgSize,K,dist,r,t);

    return {K,dist,err};
}

void saveCamera(
    const std::string& filename,
    const CameraParameters& params)
{
    cv::FileStorage fs(
        filename,
        cv::FileStorage::WRITE);

    fs << "K" << params.K;
    fs << "dist" << params.dist;
    fs.release();
}

void loadCamera(
    const std::string& filename,
    cv::Mat& K,
    cv::Mat& dist)
{
    cv::FileStorage fs(
        filename,
        cv::FileStorage::READ);

    fs["K"] >> K;
    fs["dist"] >> dist;
}
