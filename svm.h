#include <stdio.h>
#include <iostream>  
#include <fstream>  
#include <opencv2/opencv.hpp>
#include <string>
#include "LBP.h"

cv::Ptr<cv::ml::SVM> getSVM(){
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("SVM_HOG.xml");
    if (svm->empty())
    {
        std::cout << "load svm detector failed!!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    return svm;
}

float predict(cv::Mat& image, cv::Ptr<cv::ml::SVM>& svm, cv::HOGDescriptor& hog){
    cv::Mat test;
    cv::resize(image, test, cv::Size(48,48));
    //cv::cvtColor(test, test, cv::COLOR_BGR2GRAY);
    //cv::equalizeHist(test, test);

    std::vector<float> HOGdescriptors;     
    hog.compute(test, HOGdescriptors);

    cv::Mat LBP_descriptor_img;
		std::vector<float> LBP_descriptors;
		//Calcualte LBP image
    LBP_descriptors = HIST_ULBP<char>(test);

    cv::Mat testDescriptor = cv::Mat::zeros(1, HOGdescriptors.size() + LBP_descriptors.size(), CV_32FC1);

    for (size_t i = 0; i < HOGdescriptors.size(); i++)
    {
        testDescriptor.at<float>(0, i) = HOGdescriptors[i];
    }

    for (size_t i = HOGdescriptors.size(); i < HOGdescriptors.size() + LBP_descriptors.size(); i++)
    {
        testDescriptor.at<float>(0, i) = LBP_descriptors[i-HOGdescriptors.size()];
    }
    float label = svm->predict(testDescriptor);

    return label;
}

float lbp_predict(cv::Mat& image, cv::Ptr<cv::ml::SVM>& svm){
    cv::Mat test;
    cv::resize(image, test, cv::Size(48,48));
    //cv::cvtColor(test, test, cv::COLOR_BGR2GRAY);
    //cv::equalizeHist(test, test);

    cv::Mat LBP_descriptor_img;
		std::vector<float> LBP_descriptors;
		//Calcualte LBP image
		LBP_descriptors = HIST_OLBP<char>(test);

    cv::Mat testDescriptor = cv::Mat::zeros(1, LBP_descriptors.size(), CV_32FC1);


    for (size_t i = 0; i < LBP_descriptors.size(); i++)
    {
        testDescriptor.at<float>(0, i) = LBP_descriptors[i];
        //cout<<LBP_descriptors[i]<<endl;
    }
    float label = svm->predict(testDescriptor);

    return label;
}
