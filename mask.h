#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>  

using namespace std;

void generateMask(cv::Mat &inImage, cv::Mat &outMask, vector<cv::Point2f> points, vector<int> radii){
    cv::Mat mask(inImage.rows, inImage.cols, inImage.type());
    mask.setTo(cv::Scalar::all(0));

    for(int i = 0; i < points.size(); i++){
        cv::circle(mask, points[i], radii[i], (255, 255, 255), -1);
    }
    int dilation_size = 5;
    cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       cv::Point( dilation_size, dilation_size ) );

    cv::dilate(mask, mask, element, cv::Point(-1, -1), 4, 1, 1);
    cv::erode(mask, mask, element, cv::Point(-1, -1), 2);
    cv::imshow("Mask", mask);
    cv::waitKey(0);

    //cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    //cv::imshow("mask", mask);
    //cv::waitKey(0);

    outMask = mask;
}