
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

//calculates the median of input vector, the input is passed by value
template <typename T>
T median(vector<T> data)
{
    if (data.empty())
        return 0.0;

    sort(data.begin(), data.end());
    size_t index = data.size()/2;
    T value = data[index];
    if (data.size() % 2 == 0)
        value = (value + data[index-1])/2.0;
    return value;
}

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    //clear bounding box and keypoints on boundingbox
    boundingBox.keypoints.clear();
    boundingBox.kptMatches.clear();

    vector<pair<double,int>> distances;
    for (int i = 0; i < kptMatches.size(); ++i)
    {
	const auto& match = kptMatches[i];
        if (!boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
	    continue;
	double d = cv::norm(kptsCurr[match.trainIdx].pt - kptsPrev[match.queryIdx].pt);
        distances.push_back(make_pair(d, i)); 
    }
    //sort the distance based on the first element in pair
    sort(distances.begin(), distances.end());
    //retain points that are within 90% in terms of distance
    int endIdx = int(distances.size()*0.9);
    for (int i = 0; i != endIdx; ++i)
    {
	int matchIndex = distances[i].second;
	int keypointIdx = kptMatches[matchIndex].trainIdx;
        boundingBox.keypoints.push_back(kptsCurr[keypointIdx]);
	boundingBox.kptMatches.push_back(kptMatches[matchIndex]);
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dt = 1.0/frameRate;
    double lanewidth = 4.0; //width of lane

    //remove those points that doesn't belong to the ego lane
    vector<double> prev_lidar_points;
    for (auto& pt : lidarPointsPrev)
        if (fabs(pt.y) <= lanewidth/2.0)
	    prev_lidar_points.push_back(pt.x);
    vector<double> curr_lidar_points;
    for (auto& pt : lidarPointsCurr)
        if (fabs(pt.y) <= lanewidth/2.0)
	    curr_lidar_points.push_back(pt.x);
    //double prev_mean = accumulate(prev_lidar_points.begin(), 
		           //prev_lidar_points.end(), 0.0)/prev_lidar_points.size();
    //double curr_mean = accumulate(curr_lidar_points.begin(), 
		           //curr_lidar_points.end(), 0.0)/curr_lidar_points.size();
   //use median to minimize the effect of outliers
   double prev_pos = median(prev_lidar_points);
   double curr_pos = median(curr_lidar_points);

   TTC = curr_pos * dt/fabs(prev_pos - curr_pos);   
}

void getBoxIds(int index, const DataFrame& frame, std::vector<int>& box_ids)
{
    const cv::KeyPoint& keypoint = frame.keypoints[index];
    for (int i = 0; i < frame.boundingBoxes.size(); ++i)
    {
        if (frame.boundingBoxes[i].roi.contains(keypoint.pt))
        {
            box_ids.push_back(frame.boundingBoxes[i].boxID);
        }
    }
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::vector<int> curr_box_ids;
    std::vector<int> prev_box_ids;
    for (const auto& match : matches)
    {
        //find bounding box for current frame
        getBoxIds(match.trainIdx, currFrame, curr_box_ids);
        //find bounding box for previous frame
        getBoxIds(match.queryIdx, prevFrame, prev_box_ids);
    }
    //find the count for the previous/current bounding box matching combination
    std::map<int, map<int, int>> counts;
    for (auto prev_box_id : prev_box_ids)
    {
        for (auto curr_box_id : curr_box_ids)
        {
            std::map<int, int>& currmap = counts[prev_box_id];
            std::map<int, int>::iterator i = currmap.find(curr_box_id);
            currmap[curr_box_id] += 1;
        }
    }
    //find the best match bounding box
    for (int i = 0; i < prevFrame.boundingBoxes.size(); ++i)
    {
        //find the matching bounding box in current frame that has the highest count
        auto prev_box_id = prevFrame.boundingBoxes[i].boxID;
        const std::map<int, int>& currmap = counts[prev_box_id];
        auto max = std::max_element(currmap.begin(), currmap.end(), currmap.value_comp());
        bbBestMatches[prev_box_id] = max->first;
    }
}
