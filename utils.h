#ifndef UTILS
#define UTILS
#include <stdio.h>
#include <iostream>  
#include <fstream>  
#include <opencv2/opencv.hpp>

void downscaleImage(cv::Mat &inImage, cv::Mat &outImage, const float scaleFactor){
    cv::resize(inImage, outImage, cv::Size(inImage.cols * scaleFactor, inImage.rows * scaleFactor),
    0, 0, cv::INTER_LINEAR);
}

void upscaleImage(cv::Mat &inImage, cv::Mat &outImage, const float scaleFactor){
    float inverseScale = 1 / scaleFactor;
    cv::resize(inImage, outImage, cv::Size(inImage.cols * inverseScale, inImage.rows * inverseScale),
    0, 0, cv::INTER_LINEAR);
}


void upscalePoints(std::vector<cv::Point2f> &inPoints, std::vector<cv::Point2f> &outPoints, const float scaleFactor){

    float pointScale = 1 / scaleFactor;
    for(int i = 0; i < inPoints.size(); i++){
        inPoints[i].x = inPoints[i].x * pointScale;
        inPoints[i].y = inPoints[i].y * pointScale; 
    }
}

std::tuple<float, float, float> testModel(cv::Ptr<cv::ml::SVM> &svm, cv::Mat &dataFeatures, cv::Mat &dataLabels){
    int correct = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    for(int i = 0; i < dataFeatures.rows; i++){

        float label = svm->predict(dataFeatures.row(i));
        float truth = dataLabels.at<float>(i, 0);

        //cout<<label<<" "<<truth<<endl;
        
        if((label < 0 && truth < 0) || (label >= 0 && truth >= 0)){
            correct++;
        }
        else{
            if(label < 0)
                falseNegative++;

            if(label >= 0)
                falsePositive++;
        }
    }

    float accuracy = (float)correct/(float)dataFeatures.rows;
    float falsePos = (float)falsePositive/(float)dataFeatures.rows;
    float falseNeg = (float)falseNegative/(float)dataFeatures.rows;

    //cout<<"Result: "<<correct<<"/"<<dataFeatures.rows<<" Accuracy: "<<accuracy<<endl;

    return std::make_tuple(accuracy, falsePos, falseNeg);
}

#endif