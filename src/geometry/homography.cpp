#include "homography.hpp"

GroundModel computeHomography(
    const std::vector<cv::Point2f>& groundPts,
    const std::vector<cv::Point2f>& imagePts)
{
    GroundModel m;
    m.H = cv::findHomography(
            groundPts,imagePts);
    m.Hinv = m.H.inv();
    return m;
}

cv::Point2f imageToGround(
    const cv::Point2f& imgPt,
    const GroundModel& model)
{
    cv::Mat p =
        (cv::Mat_<double>(3,1)
         << imgPt.x,imgPt.y,1);

    cv::Mat g =
        model.Hinv * p;

    g /= g.at<double>(2);

    return {
        (float)g.at<double>(0),
        (float)g.at<double>(1)
    };
}

void extractPoseFromHomography(
    const cv::Mat& H,
    const cv::Mat& K,
    cv::Mat& R,
    cv::Mat& t)
{
    cv::Mat Kinv = K.inv();

    cv::Mat h1=H.col(0);
    cv::Mat h2=H.col(1);
    cv::Mat h3=H.col(2);

    cv::Mat r1 = Kinv*h1;
    cv::Mat r2 = Kinv*h2;
    t = Kinv*h3;

    double s =
        1.0 / cv::norm(r1);

    r1*=s;
    r2*=s;
    t*=s;

    cv::Mat r3 =
        r1.cross(r2);

    R=cv::Mat(3,3,CV_64F);
    r1.copyTo(R.col(0));
    r2.copyTo(R.col(1));
    r3.copyTo(R.col(2));
}
