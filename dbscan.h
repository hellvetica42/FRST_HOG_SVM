#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>

using namespace std;

const int NOISE = -2;
const int NOT_CLASSIFIED = -1;

class dbPoint {
public:
    double x, y;
    int ptsCnt, cluster;
    double getDis(const dbPoint & ot) {
        return sqrt((x-ot.x)*(x-ot.x)+(y-ot.y)*(y-ot.y));
    }
};

class DBCAN {
public:
    int n, minPts;
    double eps;
    vector<dbPoint> points;
    int size;
    vector<vector<int> > adjPoints;
    vector<bool> visited;
    vector<vector<int> > cluster;
    int clusterIdx;
    
    DBCAN(int n, double eps, int minPts, vector<dbPoint> points) {
        this->n = n;
        this->eps = eps;
        this->minPts = minPts;
        this->points = points;
        this->size = (int)points.size();
        adjPoints.resize(size);
        this->clusterIdx=-1;
    }
    void run () {
        checkNearPoints();
        
        for(int i=0;i<size;i++) {
            if(points[i].cluster != NOT_CLASSIFIED) continue;
            
            if(isCoreObject(i)) {
                dfs(i, ++clusterIdx);
            } else {
                points[i].cluster = NOISE;
            }
        }
        
        cluster.resize(clusterIdx+1);
        for(int i=0;i<size;i++) {
            if(points[i].cluster != NOISE) {
                cluster[points[i].cluster].push_back(i);
            }
        }
    }
    
    void dfs (int now, int c) {
        points[now].cluster = c;
        if(!isCoreObject(now)) return;
        
        for(auto&next:adjPoints[now]) {
            if(points[next].cluster != NOT_CLASSIFIED) continue;
            dfs(next, c);
        }
    }
    
    void checkNearPoints() {
        for(int i=0;i<size;i++) {
            for(int j=0;j<size;j++) {
                if(i==j) continue;
                if(points[i].getDis(points[j]) <= eps) {
                    points[i].ptsCnt++;
                    adjPoints[i].push_back(j);
                }
            }
        }
    }
    // is idx'th point core object?
    bool isCoreObject(int idx) {
        return points[idx].ptsCnt >= minPts;
    }
    
    vector<vector<int> > getCluster() {
        return cluster;
    }
};