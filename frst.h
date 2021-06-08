#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

#include <tuple>

#define FRST_MODE_BRIGHT 1
#define FRST_MODE_DARK 2
#define FRST_MODE_BOTH 3
#define SAMPLE_SIZE 5

using namespace std;
using namespace std::chrono;

/**
	Calculate vertical gradient for the input image

	@param input Input 8-bit image
	@param output Output gradient image
*/
void grady(const cv::Mat& input, cv::Mat &output)
{
	output = cv::Mat::zeros(input.size(), CV_64FC1);
	for (int y = 0; y<input.rows; y++)
	{
		for (int x = 1; x<input.cols - 1; x++)
		{
			*((double*)output.data + y*output.cols + x) = (double)(*(input.data + y*input.cols + x + 1) - *(input.data + y*input.cols + x - 1)) / 2;
		}
	}
}

/**
	Calculate horizontal gradient for the input image

	@param input Input 8-bit image
	@param output Output gradient image
*/
void gradx(const cv::Mat& input, cv::Mat &output)
{
	output = cv::Mat::zeros(input.size(), CV_64FC1);
	for (int y = 1; y<input.rows - 1; y++)
	{
		for (int x = 0; x<input.cols; x++)
		{
			*((double*)output.data + y*output.cols + x) = (double)(*(input.data + (y + 1)*input.cols + x) - *(input.data + (y - 1)*input.cols + x)) / 2;
		}
	}
}

/**
	Applies Fast radial symmetry transform to image
	Check paper Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for 
	detecting points of interest. Computer Vision, ECCV 2002.
	
	@param inputImage The input grayscale image (8-bit)
	@param outputImage The output image containing the results of FRST
	@param radii Gaussian kernel radius
	@param alpha Strictness of radial symmetry 
	@param stdFactor Standard deviation factor
	@param mode Transform mode ('bright', 'dark' or 'both')
*/
void frst2d(const cv::Mat& inputImage, std::vector<cv::Mat>& outputImages, std::vector<int> radii, const double alpha, const double stdFactor, const int mode) 
{
	int width = inputImage.cols;
	int height = inputImage.rows;

	cv::Mat gx, gy;
	gradx(inputImage, gx);
	grady(inputImage, gy);

	// set dark/bright mode
	bool dark = false;
	bool bright = false;

	if (mode == FRST_MODE_BRIGHT)
		bright = true;
	else if (mode == FRST_MODE_DARK)
		dark = true;
	else if (mode == FRST_MODE_BOTH) {
		bright = true;
		dark = true;
	}
	else {
		//throw std::exception("invalid mode!");		

	}


	std::vector<cv::Mat> S;
	std::vector<cv::Mat> O_n;
	std::vector<cv::Mat> M_n;

	for(int i = 0; i < radii.size(); i++){
		S.push_back(cv::Mat::zeros(inputImage.rows + 2 * radii[i], inputImage.cols + 2 * radii[i], CV_64FC1));
		O_n.push_back(cv::Mat::zeros(S[i].size(), CV_64FC1));
		M_n.push_back(cv::Mat::zeros(S[i].size(), CV_64FC1));
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {		
			cv::Point p(i, j);

			cv::Vec2d g = cv::Vec2d(gx.at<double>(i, j), gy.at<double>(i, j));

			double gnorm = std::sqrt(g.val[0] * g.val[0] + g.val[1] * g.val[1]);
			
			if (gnorm > 0) {

				cv::Vec2i gp;
				for(int i = 0; i < radii.size(); i++){
					gp.val[0] = (int)std::round((g.val[0] / gnorm) * radii[i]);
					gp.val[1] = (int)std::round((g.val[1] / gnorm) * radii[i]);
					
					if (bright) {
						cv::Point ppve(p.x + gp.val[0] + radii[i], p.y + gp.val[1] + radii[i]);

						O_n[i].at<double>(ppve.x, ppve.y) = O_n[i].at<double>(ppve.x, ppve.y) + 1;
						M_n[i].at<double>(ppve.x, ppve.y) = M_n[i].at<double>(ppve.x, ppve.y) + gnorm;
					}

					if (dark) {
						cv::Point pnve(p.x - gp.val[0] + radii[i], p.y - gp.val[1] + radii[i]);
					
						O_n[i].at<double>(pnve.x, pnve.y) = O_n[i].at<double>(pnve.x, pnve.y) - 1;
						M_n[i].at<double>(pnve.x, pnve.y) = M_n[i].at<double>(pnve.x, pnve.y) - gnorm;
					}
				}

			}
		}
	}

	double min, max;

	for(int i = 0; i < radii.size(); i++){
		O_n[i] = cv::abs(O_n[i]);
		cv::minMaxLoc(O_n[i], &min, &max);
		O_n[i] = O_n[i] / max;

		M_n[i] = cv::abs(M_n[i]);
		cv::minMaxLoc(M_n[i], &min, &max);
		M_n[i] = M_n[i] / max;

		cv::pow(O_n[i], alpha, S[i]);
		S[i] = S[i].mul(M_n[i]);

		int kSize = std::ceil(radii[i] / 2);
		if (kSize % 2 == 0)
			kSize++;

		cv::GaussianBlur(S[i], S[i], cv::Size(kSize, kSize), radii[i] * stdFactor);	
		S[i] = S[i](cv::Rect(radii[i], radii[i], width, height));

		outputImages.push_back(S[i]);
	}

	//cv::divide(1.0/radii.size(), outputImage, outputImage);
}


/**
Perform the specified morphological operation on input image with structure element of specified type and size
@param inputImage Input image of any type (preferrably 8-bit). The resulting image overwrites the input
@param operation Name of the morphological operation (MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE)
@param mShape Shape of the structure element (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
@param mSize Size of the structure element
@param iterations Number of iterations, how many times to perform the morphological operation
*/
void bwMorph(cv::Mat& inputImage, const int operation, const int mShape = cv::MORPH_RECT, const int mSize = 3, const int iterations = 1)
{
	int _mSize = (mSize % 2) ? mSize : mSize + 1;

	cv::Mat element = cv::getStructuringElement(mShape, cv::Size(_mSize, _mSize));
	cv::morphologyEx(inputImage, inputImage, operation, element, cv::Point(-1, -1), iterations);
}
/**
Perform the specified morphological operation on input image with structure element of specified type and size
@param inputImage Input image of any type (preferrably 8-bit)
@param outputImage Output image of the same size and type as the input image
@param operation Name of the morphological operation (MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE)
@param mShape Shape of the structure element (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
@param mSize Size of the structure element
@param iterations Number of iterations, how many times to perform the morphological operation
*/
void bwMorph(const cv::Mat& inputImage, cv::Mat& outputImage, const int operation, const int mShape = cv::MORPH_RECT, const int mSize = 1, const int iterations = 1)
{
	inputImage.copyTo(outputImage);

	bwMorph(outputImage, operation, mShape, mSize, iterations);
}

void NMS(cv::Mat& img, cv::Mat& outputImg, float thresh, int kSize){
	int padding = (kSize-1)/2;

	cv::Mat output(img.rows + kSize-1, img.cols + kSize-1, img.type());
	output.setTo(cv::Scalar::all(0));

	cv::Mat padded(img.rows + kSize-1, img.cols + kSize-1, img.type());
	padded.setTo(cv::Scalar::all(0));
	
	img.copyTo(padded(cv::Rect(padding, padding, img.cols, img.rows)));


	//each pixel
	for(int i = padding; i < padded.rows-padding; i++){
		for(int j = padding; j < padded.cols-padding; j++){

			//kernel
			int max = 0;
			for(int ki = i - padding; ki < i + padding; ki++){
				for(int kj = j - padding; kj < j + padding; kj++){
					int value = padded.at<char>(ki, kj); 
					if(value > max){
						max = value;
					}
				}
			}
			if(padded.at<char>(i, j) < max || padded.at<char>(i, j) < thresh){
				output.at<char>(i, j) = 0;
			}
			else{
				output.at<char>(i,j) = padded.at<char>(i,j);
			}
			/*
			if(max >= thresh){
				output.at<char>(i, j) = max;
			}
			else{
				output.at<char>(i, j) = 0;
			}
			*/
		}
	}

	outputImg = output(cv::Rect(padding, padding, img.cols, img.rows));
}

std::tuple<std::vector<cv::Point2f>, std::vector<int>> getPoints(cv::Mat& image, std::vector<int>& radiiSet, const double alpha, const double stdFactor, const int mode){
	
	//remove alpha channel
	if (image.channels() == 4) {
		cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
	}

	// convert to grayscale
//	cv::Mat grayImg;
//	cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

	//equalize histogram
//	cv::equalizeHist(grayImg, grayImg);


	//apply FRST for set of radii

	std::vector<cv::Mat> images;

	frst2d(image, images, radiiSet, alpha, stdFactor, mode);

	cv::Mat sum = cv::Mat::zeros(image.size(), CV_64FC1);

	for(int i = 0; i < images.size(); i++){
		//cv::normalize(images[i], images[i], 0.0, 255.0, cv::NORM_MINMAX);
		//sum.convertTo(images[i], CV_8U, 255.0);

		sum += images[i];
	}
	cv::normalize(sum, sum, 0.0, 255, cv::NORM_MINMAX);
	sum.convertTo(sum, CV_8UC1, 1. / radiiSet.size());

	//cv::imshow("FRST", sum);
	//cv::waitKey(0);

	//non maximal supression
	NMS(sum, sum, 15, 3);

	//cv::imshow("NMS", sum);
	//cv::waitKey(0);

	// the frst image is grayscale, let's binarize it
	cv::Mat markers;
	cv::threshold(sum, sum, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


	//cv::imshow("BINARY", sum);
	//cv::waitKey(0);

	bwMorph(sum, markers, cv::MORPH_CLOSE, cv::MORPH_ELLIPSE, 5, 2);
	//bwMorph(sum, markers, cv::MORPH_OPEN, cv::MORPH_ELLIPSE, 1, 5);

	//cv::imshow("BWMORPH", markers);
	//cv::waitKey(0);

	// the 'markers' image contains dots of different size. Let's vectorize it
	std::vector< std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	contours.clear();
	cv::findContours(markers, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// get the moments
	std::vector<cv::Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	//  get the mass centers and radius of points:
	std::vector<int> pointRadii(contours.size());
	std::vector<cv::Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = cv::Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

		if(isnan(mc[i].x) || isnan(mc[i].y)){
			pointRadii[i] = 2;
			continue;
		}

		int maxRadius = 0;
		int maxIndex = 0;
		for(int s = 0; s < radiiSet.size(); s++){

			int value = images[s].at<char>(mc[i]);

			if(value > maxRadius){
				maxRadius = value;
				maxIndex = s;
			}
		}

		pointRadii[i] = radiiSet[maxIndex];
	}

	return std::make_tuple(mc, pointRadii);
}
