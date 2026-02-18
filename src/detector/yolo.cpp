#include "yolo.hpp"

YOLODetector::YOLODetector(const std::string& modelPath,
                           float confThreshold,
                           float nmsThreshold)
    : confThreshold(confThreshold),
      nmsThreshold(nmsThreshold)
{
    net = cv::dnn::readNet(modelPath);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& frame)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame,
                           blob,
                           1.0/255.0,
                           cv::Size(inputWidth, inputHeight),
                           cv::Scalar(),
                           true,
                           false);

    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat output = outputs[0];

    const int dimensions = 85;
    const int rows = output.size[1];

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* data = (float*)output.data;

    for (int i = 0; i < rows; ++i)
    {
        float obj_conf = data[4];

        if (obj_conf < confThreshold)
        {
            data += dimensions;
            continue;
        }

        cv::Mat scores(1, 80, CV_32FC1, data + 5);
        cv::Point classIdPoint;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

        double confidence = obj_conf * maxClassScore;

        if (confidence >= confThreshold &&
            classIdPoint.x == 2) // car class in COCO
        {
            float cx = data[0];
            float cy = data[1];
            float w  = data[2];
            float h  = data[3];

            int left   = int((cx - 0.5f * w) * frame.cols / inputWidth);
            int top    = int((cy - 0.5f * h) * frame.rows / inputHeight);
            int widthB = int(w * frame.cols / inputWidth);
            int heightB= int(h * frame.rows / inputHeight);

            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)confidence);
            boxes.push_back(cv::Rect(left, top, widthB, heightB));
        }

        data += dimensions;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes,
                      confidences,
                      confThreshold,
                      nmsThreshold,
                      indices);

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        detections.push_back({
            classIds[idx],
            confidences[idx],
            boxes[idx]
        });
    }

    return detections;
}
