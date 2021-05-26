#include "frst.h"
#include "utils.h"

#include <iostream>
#include <string.h>
#include <filesystem>
#include <fstream>

#define RADII 20
#define STDDEV 0.2
#define MODE 3
#define ALPHA 1.3

#define IMG_SIZE 50

#define PADDING 5

#define SCALE_FACTOR 1
#define ORIG_DOWNSAMPLE 0.5

using namespace std;

std::size_t number_of_files_in_directory(std::filesystem::path path)
{
    using std::filesystem::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}


int main(int argc, char* argv[]) {
	cv::Mat image;
	cv::Mat originalImage;

	if (argc > 1) {
		originalImage = cv::imread(argv[1]);
	}
	else {
		std::cout << "Image not specified" << std::endl;
		return -1;
	}

	if (!originalImage.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	downscaleImage(originalImage, originalImage, ORIG_DOWNSAMPLE);	
	downscaleImage(originalImage, image, SCALE_FACTOR);	

	//cv::pyrDown(image, image, cv::Size(image.cols/2, image.rows/2));
	cv::Mat image_marked = image.clone();

	cv::Mat grayImg;
	cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

	//equalize histogram
	cv::equalizeHist(grayImg, grayImg);

	std::vector<cv::Point2f> points;

	std::vector<int> radiiSet = {5, 7, 9 };
	std::vector<int> radii;

	tie(points, radii) = getPoints(grayImg, radiiSet, ALPHA, STDDEV, MODE);

	cv::Mat cropped;
	int posCount = number_of_files_in_directory("data/positive");
	int negCount = number_of_files_in_directory("data/negative");
	int testCount = number_of_files_in_directory("data/test");

	for(int i = 0; i<points.size(); i++){
			cv::circle(image_marked, points[i], radii[i], CV_RGB(0, 255, 0), 1, 8);
	}
	for(int i = 0; i<points.size(); i++){
		int radius = (radii[i] + PADDING);

		if(points[i].x < radius || points[i].y < radius || isnan(points[i].x) || isnan(points[i].y)){
			continue;
		}

		cv::Rect roi((int)points[i].x-radius, (int)points[i].y-radius, radius*2, radius*2);

		if(roi.x >= 0 && roi.y >= 0 && roi.width + roi.x < image.cols && roi.height + roi.y < image.rows){
			cropped = image(roi);

			cv::resize(cropped, cropped, cv::Size(IMG_SIZE, IMG_SIZE), 0, 0, cv::INTER_LINEAR);

			cv::Point2f pt = (1/SCALE_FACTOR) * points[i];

			cv::imshow("Cropped", cropped);
			cv::imshow("Image", image_marked);
			char ch = cv::waitKey(0);
			
			if(ch == 'g'){
				cv::imwrite("data/positive/pos" + to_string(posCount++) + ".jpg", cropped);
			}
			else if(ch == 'b'){
				cv::imwrite("data/negative/neg" + to_string(negCount++) + ".jpg", cropped);
			}
			else if(ch == 't'){
				cv::imwrite("data/test/test" + to_string(testCount++) + ".jpg", cropped);
			}
			else if(ch == 'n'){
				continue;
			}

		}
	}
}
