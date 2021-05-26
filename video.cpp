#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

#include "grapeMask.h"
#include "utils.h"

using std::string;
using namespace std;

int main(int argc, char* argv[]){
    string filename;

    if (argc > 1){
       filename = argv[1] ;
    }
    else {
        cout<<"No file specified"<<endl;
        return 1; 
    }

    cv::VideoCapture capture(filename);
    cv::Mat frame;

    if (!capture.isOpened())
        throw "Error reading video file";

    while(true){
        capture >> frame;
        if(frame.empty())
            break;


        downscaleImage(frame, frame, 0.7);
        getGrapeImage(frame, frame);

        cv::imshow("video", frame);

        char key = cv::waitKey(0);
        if(key == 'q')
            break;
    }
        
}