#include "frst.h"
#include "svm.h"
#include "utils.h"

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

#define RADII 20
#define ALPHA 0.5
#define STDDEV 0.5
#define MODE 3

#define IMG_SIZE 50
#define PADDING 5

#define SCALE_FACTOR 1
#define ORIG_DOWNSAMPLE 0.5

int main(int argc, char* argv[]) {
	cv::Mat image;
	cv::Mat originalImage;

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

	downscaleImage(originalImage, originalImage, ORIG_DOWNSAMPLE);	
	downscaleImage(originalImage, image, SCALE_FACTOR);	

	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

	//equalize histogram
	cv::equalizeHist(grayImg, grayImg);

	cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	cv::Ptr<cv::ml::SVM> svm = getSVM();

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	std::vector<cv::Point2f> points;
	std::vector<int> radiiSet = {11, 13, 15};
	//std::vector<int> radiiSet = {5, 7, 11, 13};
	std::vector<int> radii;

	tie(points, radii) = getPoints(grayImg, radiiSet, ALPHA, STDDEV, FRST_MODE_BOTH);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	cout<<"frst took "<<time_span.count()<<" seconds"<<endl;


	cv::Mat cropped;
	for(int i = 0; i < points.size(); i++){
		int radius = radii[i] + PADDING;

		if(points[i].x < radius || points[i].y < radius || isnan(points[i].x) || isnan(points[i].y)){
			continue;
		}

		cv::Rect roi((int)points[i].x-radius, (int)points[i].y-radius, radius*2, radius*2);

		if(roi.x >= 0 && roi.y >= 0 && roi.width + roi.x < image.cols && roi.height + roi.y < image.rows){
			cropped = grayImg(roi);

			cv::equalizeHist(cropped, cropped);
			//cv::resize(cropped, cropped, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::INTER_LINEAR);

			float cl = predict(cropped, svm, hog);
			//cout<<cl<<endl;
			cv::Point2f pt = (1/SCALE_FACTOR) * points[i];
			int rad = (1/SCALE_FACTOR) * radius;
			if(cl > 0){
				cv::circle(originalImage, pt, rad, CV_RGB(0, 255, 0), 1, 8, 0);
			}
			else{
				cv::circle(originalImage, pt, 2, CV_RGB(255, 0, 0), 1, 8, 0);
			}
		}

	}


	//upscaleImage(image, image, SCALE_FACTOR);
	cv::imshow("Image", originalImage);
	cv::waitKey(0);
}
