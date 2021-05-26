#include <stdio.h>
#include <iostream>  
#include <fstream>  
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

#include "LBP.h"
#include "utils.h"
 
using namespace std::chrono;
using namespace cv::ml;
using namespace std;
 
#define PosSamNO 1152 //Number of positive samples                                                    
#define NegSamNO 1152 //Number of negative samples                                     
#define Validation 500//Number of tests                                                    
#define K 10

void split_data_K(cv::Mat &dataFeaturesMat, cv::Mat &dataLabelsMat,
				  cv::Mat &outTrainFeatures, cv::Mat &outTrainLabels,
				  cv::Mat &outTestFeatures, cv::Mat &outTestLabels, int K_num, int K_index){


	int containerSize = dataFeaturesMat.rows / K_num;

	for(int i = 0; i < dataFeaturesMat.rows; i++){
		if(i > K_index * containerSize && i < (K_index+1) * containerSize){
			outTestFeatures.push_back(dataFeaturesMat.row(i));
			outTestLabels.push_back(dataLabelsMat.row(i));
		}
		else{
			outTrainFeatures.push_back(dataFeaturesMat.row(i));
			outTrainLabels.push_back(dataLabelsMat.row(i));
		}
	}

}

int main(int argc, char** argv){

	 //HOG detector, used to calculate the HOG descriptor
	 //Detection window (48,48), block size (16,16), block step size (8,8), cell size (8,8), histogram bin number 9 
	cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
	 int HOGDescriptorDim;//The dimension of the HOG descriptor is determined by the size of the picture, the size of the detection window, the block size, and the number of bins in the histogram of the cell unit  
	 int LBPDescriptorDim;
 
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
            std::cout<<"Image " << ImgName << "is empty"<<endl;
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
		} //The positive sample category is 1, judged as no splash	 

		//Copy the LBP descriptor to featureMatrix
		for (int i = HOGDescriptorDim; i < LBPDescriptorDim + HOGDescriptorDim; i++)
		{
			 //The ith element in the feature vector of the num sample 
			sampleFeatureMat.at<float>(num, i) = LBP_descriptors[i-HOGDescriptorDim];
		} //The positive sample category is 1, judged as no splash	 
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

    cv::Mat shuffledFeatureMat;
    cv::Mat shuffledLabelMat;

    cv::Mat validationFetureMat;
    cv::Mat validationLabelMat;

    std::cout<<"Sample feature mat: "<<sampleFeatureMat.size()<<endl;
    std::cout<<"Sample label mat: "<<sampleLabelMat.size()<<endl;

    //shuffle the data (mix positive and negative)
    std::vector<int> seeds;
    for (int cont = 0; cont < sampleFeatureMat.rows; cont++){
        seeds.push_back(cont);
    }

    cv::randShuffle(seeds);

    for(int cont = 0; cont < sampleFeatureMat.rows; cont++){
        shuffledFeatureMat.push_back(sampleFeatureMat.row(seeds[cont]));
        shuffledLabelMat.push_back(sampleLabelMat.row(seeds[cont]));
    }
    std::cout<<"Shuffled the data"<<endl;


    const float gammaLow = 0.05;
    const float gammaHigh = 0.25;
    const float cLow = 1;
    const float cHigh = 20;

	const float gammaInterval = 0.01;
	const float cInterval = 1;

	const float mul = 10;


	float maxAccuracy = 0;
	float maxGamma = 0;
	float maxC = 0;

	ofstream logfile;
	logfile.open("logs/005_1_to_025_20_1000MAXITER_1e-6EPS_T-03.txt");
    for(float gamma = gammaLow; gamma <= gammaHigh; gamma+=gammaInterval){
        for(float c = cLow; c <= cHigh; c+=cInterval){
			std::cout<<"Training with paramters: Gamma: "<<gamma<<" C: "<<c;
			logfile << "G:"<<gamma<<" C:"<<c;

			float accumulate_accuracy = 0;
			float accumulate_falsePos = 0;
			float accumulate_falseNeg = 0;
			//run all K folds
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
			for(int i = 0; i < K; i++){

				//Set SVM parameters	
				cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
				svm->setType(cv::ml::SVM::Types::C_SVC);
				svm->setKernel(cv::ml::SVM::KernelTypes::RBF);
				//svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS, 200, 1e-3));
				svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 
														1000, 1e-6));
				svm->setGamma(gamma);
				svm->setC(c);

				cv::Mat trainFeatures;
				cv::Mat trainLabels;

				cv::Mat testFeatures;
				cv::Mat testLabels;

				split_data_K(shuffledFeatureMat, shuffledLabelMat,
							 trainFeatures, trainLabels,
							 testFeatures, testLabels, K, i);


				//Train SVM classifier  
				cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainFeatures,
																		 cv::ml::SampleTypes::ROW_SAMPLE,
																		 trainLabels);

				svm->train(td);
				float acc, fPos, fNeg;
				tie(acc, fPos, fNeg) = testModel(svm, testFeatures, testLabels);
				accumulate_accuracy += acc;
				accumulate_falsePos += fPos;
				accumulate_falseNeg += fNeg;
			}

			float accuracy = accumulate_accuracy/K;
			if(accuracy > maxAccuracy){
				maxAccuracy = accuracy;
				maxGamma = gamma;
				maxC = c;
			}

			float falsePositive = accumulate_falsePos/K;
			float falseNegative = accumulate_falseNeg/K;

			std::cout<<" Accuracy: "<<accuracy<<" fPositive: "<<falsePositive<<" fNegative: "<<falseNegative;
			logfile<<" A:"<<accuracy<<" FPOS:"<<falsePositive<<" FNEG:"<<falseNegative<<endl;;

			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			std::cout<<" took: "<<time_span.count()<<" seconds"<<endl;

            //cout<<"Result: "<<acc*Validation<<"/"<<Validation<<" Accuracy: "<<acc<<endl;
        }
    }

	std::cout<<"Best accuracy: "<<maxAccuracy<<" with parameters Gamma: "<<maxGamma<<" C: "<<maxC;
 
	logfile.close();
    return 0;
}