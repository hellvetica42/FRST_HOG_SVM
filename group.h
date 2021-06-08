#ifndef GROUP 
#define GROUP

#include "dbscan.h"

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

using namespace std;

//epsilon -- max distance between points
//minPoints -- minimum points in a cluster
void groupPoints(vector<cv::Point2f> &inPoints, vector<vector<int>> &outCluster, int epsilon, int minPoints){

    vector<dbPoint> dbPoints;
    for(int i = 0; i < inPoints.size(); i++){
        dbPoints.push_back({inPoints.at(i).x, inPoints.at(i).y, 0, NOT_CLASSIFIED});
    }

    DBCAN dbScan(dbPoints.size(), epsilon, minPoints, dbPoints);
    dbScan.run();

    outCluster = dbScan.getCluster();
}

#endif