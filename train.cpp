#include <stdio.h>
#include <iostream>  
#include <fstream>  
#include <opencv2/opencv.hpp>
#include <string>

#include "LBP.h"
#include "utils.h"
 
using namespace cv::ml;
using namespace std;
 
#define PosSamNO 1960 //Number of positive samples                                                    
 #define NegSamNO 1760 //Number of negative samples                                     
 #define TestSamNO 28 //Number of tests                                                    
 
void train_svm_hog()
{
 
	 //HOG detector, used to calculate the HOG descriptor
	 //Detection window (48,48), block size (16,16), block step size (8,8), cell size (8,8), histogram bin number 9 
	cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	 int HOGDescriptorDim;//The dimension of the HOG descriptor is determined by the size of the picture, the size of the detection window, the block size, and the number of bins in the histogram of the cell unit  
	 int LBPDescriptorDim;
 
	 //Set SVM parameters	
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::Types::C_SVC);
	svm->setKernel(cv::ml::SVM::KernelTypes::RBF);
	svm->setC(2);
	svm->setGamma(0.15);
	//svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 350, 1e-6));
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));
	/*
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setGamma(0);
	svm->setNu(0.5);
	svm->setP(0.1);
	svm->setType(cv::ml::SVM::Types::EPS_SVR);
	*/
	std::string ImgName;
 
	 //File list of positive sample pictures
	std::ifstream finPos("data/positive_samples.txt");
	 //File list of negative sample pictures
	std::ifstream finNeg("data/negative_samples.txt");
 
	 //A matrix of feature vectors of all training samples, the number of rows is equal to the number of all samples, the number of columns is equal to the dimension of the HOG descriptor 
	cv::Mat sampleFeatureMat;
	 //Category vector of training samples, the number of rows is equal to the number of all samples, the number of columns is equal to 1; 1 means there is a target, -1 means no target 
	cv::Mat sampleLabelMat;

	cv::Mat LBP_descriptor_img;
 
	 //Read positive sample images in turn to generate HOG descriptors  
	for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
	{
		std::cout << "Processing：" << ImgName << std::endl;
		cv::Mat image = cv::imread(ImgName, cv::IMREAD_COLOR);
        if(image.empty()){
            cout<<"Image " << ImgName << "is empty"<<endl;
            continue;
        }
		cv::resize(image, image, cv::Size(48, 48));
		cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(image, image);
 
		 //HOG descriptor vector
		std::vector<float> descriptors;
		 //Calculate the HOG descriptor and detect the moving step of the window (8,8)
		hog.compute(image, descriptors, cv::Size(8, 8));

		//LBP descriptor vector
		std::vector<float> LBP_descriptors;
		//Calcualte LBP image histogram
		LBP_descriptors = HIST_ULBP<char>(image);

		 //Initialize the eigenvector matrix and category matrix when processing the first sample, because the eigenvector matrix can only be initialized if the dimension of the eigenvector is known 
		if (0 == num)
		{
			 //Dimension of HOG descriptor
			HOGDescriptorDim = descriptors.size();

			//Dimension of LBP descriptor
			LBPDescriptorDim = LBP_descriptors.size();
			 //Initialize the matrix of feature vectors of all training samples, the number of rows is equal to the number of all samples, the number of columns is equal to the dimension of HOG descriptor sub sampleFeatureMat 
			sampleFeatureMat = cv::Mat::zeros(PosSamNO + NegSamNO, HOGDescriptorDim + LBPDescriptorDim, CV_32FC1);
			 //Initialize the category vector of training samples, the number of rows is equal to the number of all samples, the number of columns is equal to 1
			sampleLabelMat = cv::Mat::zeros(PosSamNO + NegSamNO, 1, CV_32SC1);
		}

		 //Copy the calculated HOG descriptor to the sample feature matrix sampleFeatureMat  
		for (int i = 0; i < HOGDescriptorDim; i++)
		{
			 //The ith element in the feature vector of the num sample 
			sampleFeatureMat.at<float>(num, i) = descriptors[i];
		} 

		//Copy the LBP descriptor to featureMatrix				
		for (int i = HOGDescriptorDim; i < LBPDescriptorDim + HOGDescriptorDim; i++)
		{
			 //The ith element in the feature vector of the PosSamNO+num samples
			sampleFeatureMat.at<float>(num, i) = LBP_descriptors[i-HOGDescriptorDim];
		}

		//The positive sample category is 1, judged as no splash	 
		sampleLabelMat.at<float>(num, 0) = 1;

	}
 
 
	 //Read negative sample pictures in turn to generate HOG descriptors  
	for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
	{
		std::cout << "Processing：" << ImgName << std::endl;
		cv::Mat src = cv::imread(ImgName);

		cv::resize(src, src, cv::Size(48, 48));
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(src, src);
 
		 //HOG descriptor vector		
		std::vector<float> descriptors;
		 //Calculate the HOG descriptor and detect the moving step of the window (8,8) 
		hog.compute(src, descriptors, cv::Size(8, 8));

		//LBP descriptor vector
		std::vector<float> LBP_descriptors;
		//Calcualte LBP image
		LBP_descriptors = HIST_ULBP<char>(src);

		 //Initialize the eigenvector matrix and category matrix when processing the first sample, because the eigenvector matrix can only be initialized if the dimension of the eigenvector is known 
		//std::cout << "descriptor dimention：" << descriptors.size() << std::endl;
 
		 //Copy the calculated HOG descriptor to the sample feature matrix sampleFeatureMat
		for (int i = 0; i < HOGDescriptorDim; i++)
		{
			 //The ith element in the feature vector of the PosSamNO+num samples
			sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];
		}

		for (int i = HOGDescriptorDim; i < LBPDescriptorDim + HOGDescriptorDim; i++)
		{
			 //The ith element in the feature vector of the PosSamNO+num samples
			sampleFeatureMat.at<float>(num + PosSamNO, i) = LBP_descriptors[i-HOGDescriptorDim];
		}
		 //Negative sample category is -1, which is judged as splashing
		sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;
	}

	cv::Mat bkpFeature = sampleFeatureMat.clone();
	cv::Mat bkpLabel = sampleLabelMat.clone();
 
	 //Train SVM classifier  
	std::cout << "Start training SVM classifier" << std::endl;
	cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(sampleFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);

	// svm->trainAuto(td, 10, SVM::getDefaultGrid(SVM::C), 
	// 					SVM::getDefaultGrid(SVM::GAMMA), 
	// 					SVM::getDefaultGrid(SVM::P),
	// 					SVM::getDefaultGrid(SVM::NU),
	// 					SVM::getDefaultGrid(SVM::COEF),
	// 					SVM::getDefaultGrid(SVM::DEGREE),
	// 					true);
	svm->train(td);
	std::cout << "SVM classifier training completed" << std::endl;

	float acc, fPos, fNeg;
	tie(acc, fPos, fNeg) = testModel(svm, bkpFeature, bkpLabel);
	cout<<acc<<endl;
 
	 //Save the trained SVM model as an xml file
	svm->save("SVM_HOG.xml");
	return;
}
 
void svm_hog_classification()
{
	 //HOG detector, used to calculate the HOG descriptor
	 //Detection window (48,48), block size (16,16), block step size (8,8), cell size (8,8), histogram bin number 9  
	cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	 //The dimension of the HOG descriptor is determined by the image size, detection window size, block size, and the number of histogram bins in the cell unit 
	int DescriptorDim;
 
	 //File list of test sample pictures
	std::ifstream finTest("data/test_samples.txt");
	std::string ImgName;
	for (int num = 0; num < TestSamNO && getline(finTest, ImgName); num++)
	{
		 //Read the trained SVM model from the XML file
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("SVM_HOG.xml");
		if (svm->empty())
		{
			std::cout << "load svm detector failed!!!" << std::endl;
			return;
		}
		 //Identify the test set
		 std::cout << "Start recognition..." << std::endl;
		std::cout << "Processing：" << ImgName << std::endl;
		cv::Mat test = cv::imread(ImgName);
		cv::resize(test, test, cv::Size(48, 48));

		cv::cvtColor(test, test, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(test, test);
		std::vector<float> descriptors;     

		hog.compute(test, descriptors, cv::Size(8,8));

		//LBP descriptor vector
		std::vector<float> LBP_descriptors;
		//Calcualte LBP image histogram
		LBP_descriptors = HIST_ULBP<char>(test);

		cv::Mat testDescriptor = cv::Mat::zeros(1, descriptors.size() + LBP_descriptors.size(), CV_32FC1);
		for (size_t i = 0; i < descriptors.size(); i++)
		{
			testDescriptor.at<float>(0, i) = descriptors[i];
		}

		for(size_t i = descriptors.size(); i < LBP_descriptors.size(); i++){
			testDescriptor.at<float>(0, i) = LBP_descriptors[i];
		}
		float label = svm->predict(testDescriptor);
		imshow("test image", test);
		 std::cout << "This picture belongs to:" << label << std::endl;
		cv::waitKey(0);
	}
	return;
}
 
int main(int argc, char** argv)
{
	train_svm_hog();
	//svm_hog_classification();
	return 0;
}
