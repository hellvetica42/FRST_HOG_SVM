#include "frst.h"
#include "svm.h"
#include "utils.h"
#include "group.h"
#include "mask.h"

#include <chrono>
#include <iostream>
#include <string.h>
#include <fstream>

using namespace std::chrono;
using namespace cv::ml;
using namespace std;

#define FRST_MODE_BRIGHT 1
#define FRST_MODE_DARK 2
#define FRST_MODE_BOTH 3

#define ALPHA 1.2
#define STDDEV 0.2

#define MIN_NEIGHBOURS 3
#define MIN_CLUSTER_SIZE 4
#define MAX_DISTANCE 35

#define IMG_SIZE 50
#define PADDING 0.1

#define SCALE_FACTOR 1
#define ORIG_DOWNSAMPLE 0.5

int main(int argc, char* argv[]) {
	cv::Mat image;
	cv::Mat originalImage;
	cv::Mat finalImage;

	if (argc > 1) {
		originalImage = cv::imread(argv[1]);
	}
	else {
		originalImage = cv::imread("image.jpeg");
	}

	if (!originalImage.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	downscaleImage(originalImage, originalImage, ORIG_DOWNSAMPLE);	
	finalImage = originalImage.clone();
	downscaleImage(originalImage, image, SCALE_FACTOR);	

	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

	//equalize histogram
	cv::equalizeHist(grayImg, grayImg);

	cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	cv::Ptr<cv::ml::SVM> svm = getSVM();


	std::vector<cv::Point2f> points;
	//std::vector<int> radiiSet = {11, 13, 15};
	std::vector<int> radiiSet = {5, 7, 9};
	std::vector<int> radii;

	tie(points, radii) = getPoints(grayImg, radiiSet, ALPHA, STDDEV, FRST_MODE_BRIGHT);


	cv::Mat cropped;
	std::vector<cv::Point2f> validPoints;
	std::vector<int> validRadii;
	for(int i = 0; i < points.size(); i++){
		int radius = radii[i] + radii[i] * PADDING;

		if(points[i].x < radius || points[i].y < radius || isnan(points[i].x) || isnan(points[i].y)){
			continue;
		}

		cv::Rect roi((int)points[i].x-radius, (int)points[i].y-radius, radius*2, radius*2);

		if(roi.x >= 0 && roi.y >= 0 && roi.width + roi.x < image.cols && roi.height + roi.y < image.rows){
			cropped = grayImg(roi);


			//cv::equalizeHist(cropped, cropped);
			//cv::resize(cropped, cropped, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::INTER_LINEAR);

			float cl = predict(cropped, svm, hog);
			//cout<<cl<<endl;
			cv::Point2f pt = (1/SCALE_FACTOR) * points[i];
			int rad = (1/SCALE_FACTOR) * radius;
			if(cl > 0){
				validPoints.push_back(points[i]);
				validRadii.push_back(radii[i]);
				cv::circle(originalImage, pt, 2, CV_RGB(0, 255, 0), 3, 8, 0);
			}
			else{
				cv::circle(originalImage, pt, 2, CV_RGB(255, 0, 0), 1, 8, 0);
			}
		}

	}

	//cluster points
	vector<vector<int>> clusters;
	groupPoints(validPoints, clusters, MAX_DISTANCE, MIN_NEIGHBOURS);

	//std::cout<<clusters.size()<<std::endl;

	std::vector<cv::Point2f> finalPoints;
	std::vector<int> finalRadii;

	for(int c = 0; c < clusters.size(); c++){

		if(clusters.at(c).size() < MIN_CLUSTER_SIZE)
			continue;


		cv::Scalar color = cv::Scalar(0, 0, 255);

		for(int p = 0; p < clusters.at(c).size(); p++){
			cv::Point2f pt = (1/SCALE_FACTOR) * validPoints[clusters.at(c).at(p)];
			int rad = (1/SCALE_FACTOR) * validRadii[clusters.at(c).at(p)];
			cv::circle(originalImage, pt, rad, color, 2, 8, 0);

			finalPoints.push_back(validPoints[clusters.at(c).at(p)]);
			finalRadii.push_back(validRadii[clusters.at(c).at(p)]);
		}

	}

	cv::Mat mask;
	generateMask(grayImg, mask, finalPoints, finalRadii);

	cv::Mat cannyOutput;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size(); i++){
		cv::drawContours(finalImage, contours, i, cv::Scalar(0, 0, 255), 2, cv::LINE_8, hierarchy, 0);
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout<<"took "<<time_span.count()<<" seconds"<<endl;


	//upscaleImage(image, image, SCALE_FACTOR);
	cv::imshow("Data", originalImage);
	cv::imshow("Contours", finalImage);
	cv::waitKey(0);
}
