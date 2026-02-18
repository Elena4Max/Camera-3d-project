#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "calibration/calibration.hpp"
#include "detector/yolo.hpp"
#include "geometry/homography.hpp"
#include "geometry/projection.hpp"

namespace fs = std::filesystem;

std::vector<std::string> getImages(const std::string& folder)
{
    std::vector<std::string> paths;
    for (auto& p : fs::recursive_directory_iterator(folder))
    {
        if (p.path().extension() == ".jpg" ||
            p.path().extension() == ".png")
        {
            paths.push_back(p.path().string());
        }
    }
    return paths;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage:\n";
        std::cout << "  app calibrate <image_folder>\n";
        std::cout << "  app demo <video> <camera.yaml>\n";
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "calibrate")
    {
        std::string folder = argv[2];

        auto images = getImages(folder);

        CameraParameters params =
            runCalibration(images,
                           cv::Size(9,6),
                           0.025f);

        saveCamera("camera.yaml", params);

        std::cout << "Reprojection error: "
                  << params.reprojectionError
                  << std::endl;
    }

    if (mode == "demo")
    {
        std::string videoPath = argv[2];
        std::string cameraFile = argv[3];

        cv::Mat K, dist;
        loadCamera(cameraFile, K, dist);

        YOLODetector detector("models/yolov5n.onnx");

        cv::VideoCapture cap(videoPath);

        cv::Mat frame;
        cap >> frame;

        std::vector<cv::Point2f> groundPts =
        {
            {0,0},
            {3.5f,0},
            {3.5f,20},
            {0,20}
        };

        std::vector<cv::Point2f> imagePts =
        {
            {200,720},
            {1080,720},
            {780,350},
            {500,350}
        };

        GroundModel ground =
            computeHomography(groundPts, imagePts);

        cv::Mat R, t;
        extractPoseFromHomography(
            ground.H, K, R, t);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        cv::VideoWriter writer(
            "output.mp4",
            cv::VideoWriter::fourcc('m','p','4','v'),
            30,
            frame.size());

        while (cap.read(frame))
        {
            cv::undistort(frame, frame, K, dist);

            auto detections =
                detector.detect(frame);

            for (const auto& det : detections)
            {
                float u = det.box.x +
                          det.box.width/2.0f;
                float v = det.box.y +
                          det.box.height;

                cv::Point2f g =
                    imageToGround({u,v}, ground);

                draw3DBox(frame,
                          K, dist, R, t,
                          g.x, g.y,
                          1.8f,4.2f,1.5f);
            }

            writer.write(frame);
        }
    }

    return 0;
}
