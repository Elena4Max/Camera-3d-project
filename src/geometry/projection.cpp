#include "projection.hpp"

void draw3DBox(
    cv::Mat& frame,
    const cv::Mat& K,
    const cv::Mat& dist,
    const cv::Mat& R,
    const cv::Mat& t,
    float X,float Y,
    float w,float l,float h)
{
    std::vector<cv::Point3f> pts =
    {
        {X-w/2,Y,0},
        {X+w/2,Y,0},
        {X+w/2,Y+l,0},
        {X-w/2,Y+l,0},

        {X-w/2,Y,h},
        {X+w/2,Y,h},
        {X+w/2,Y+l,h},
        {X-w/2,Y+l,h}
    };

    cv::Mat rvec;
    cv::Rodrigues(R,rvec);

    std::vector<cv::Point2f> img;

    cv::projectPoints(
        pts,rvec,t,K,dist,img);

    for(int i=0;i<4;i++)
        cv::line(frame,img[i],
                 img[(i+1)%4],
                 {0,255,0},2);

    for(int i=4;i<8;i++)
        cv::line(frame,img[i],
                 img[4+(i+1)%4],
                 {0,255,0},2);

    for(int i=0;i<4;i++)
        cv::line(frame,img[i],
                 img[i+4],
                 {0,255,0},2);
}
