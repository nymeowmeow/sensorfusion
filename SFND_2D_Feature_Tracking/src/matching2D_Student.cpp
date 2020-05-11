#include <numeric>
#include "matching2D.hpp"
#include <stdexcept>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, 
		     std::vector<cv::KeyPoint> &kPtsRef, 
		     cv::Mat &descSource, cv::Mat &descRef,
                     std::vector<cv::DMatch> &matches, 
		     std::string descriptorType, 
		     std::string matcherType, 
		     std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType == "DES_BINARY" ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
	{
	    //OpenCV buf workaround: convert binary descriptors to floating point due to a bug in current OpenCV implementation
	    descSource.convertTo(descSource, CV_32F);
	    descRef.convertTo(descRef, CV_32F);
	}

	matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } else {
        //raise exception
	throw invalid_argument("invalid matcher type: " + matcherType);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
	matcher->knnMatch(descSource, descRef, knn_matches, 2); //find the 2 best matches

	//filter matches using descriptor distance ratio test
	double minDescDistRatio = 0.8;
	for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
	{
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
	    {
	        matches.push_back((*it)[0]);
	    }
	}
    } else {
        //raise exception
	throw invalid_argument("invalid matcher selection type: " + selectorType);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double 
descKeypoints(vector<cv::KeyPoint> &keypoints, 
	      cv::Mat &img, cv::Mat &descriptors, 
	      string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType == "BRIEF") {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    } else if (descriptorType == "ORB") {
	extractor = cv::ORB::create();
    } else if (descriptorType == "FREAK") {
        extractor = cv::xfeatures2d::FREAK::create();
    } else if (descriptorType == "AKAZE") {
        extractor = cv::AKAZE::create();
    } else if (descriptorType == "SIFT") {
        extractor = cv::xfeatures2d::SIFT::create();
    } else {
        throw invalid_argument("invalid descriptor type: " + descriptorType);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double 
detKeypoints(const string& detectorType, 
	     vector<cv::KeyPoint> &keypoints, 
	     cv::Mat &img, bool bVis)
{
    double t = 0;
    if (detectorType == "SHITOMASI")
    {
        // compute detector parameters based on image size
        int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
        double maxOverlap = 0.0; // max. permissible overlap between two features in %
        double minDistance = (1.0 - maxOverlap) * blockSize;
        int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

        double qualityLevel = 0.01; // minimal accepted quality of image corners
        double k = 0.04;

        // Apply corner detection
	t = (double)cv::getTickCount();
        vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(img, corners, maxCorners, 
	        qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

        // add corners to result vector
        for (auto it = corners.begin(); it != corners.end(); ++it)
        {
    
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
            newKeyPoint.size = blockSize;
            keypoints.push_back(newKeyPoint);
        }
    } else if (detectorType == "HARRIS") {
        //detector parameters
	int blockSize = 2; //for every pixel, a blockSize x blockSize neighborhood is considered
	int apertureSize = 3; //aperture parameter for Sobel operator (must be odd)
	int minResponse = 100; //minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;

	//detect harris corners and normalize output
	t = (double)cv::getTickCount();

	cv::Mat dst, dst_norm, dst_norm_scaled;
	dst = cv::Mat::zeros(img.size(), CV_32FC1);
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs(dst_norm, dst_norm_scaled);

	double maxOverlap = 0.0; //max. permissible overlap between two features in %, used during non-maxima suppression
	for (size_t j = 0; j < dst_norm.rows; ++j)
	{
            for (size_t i = 0; i < dst_norm.cols; ++i)
	    {
                int response = (int)dst_norm.at<float>(j, i);
		if (response > minResponse)
		{
                    //only store points above a threshold
		    cv::KeyPoint newKeyPoint;
		    newKeyPoint.pt = cv::Point2f(i, j);
		    newKeyPoint.size = 2*apertureSize;
		    newKeyPoint.response = response;

		    //perform non-maximum suppression (NMS) in local neighourhood around new key point
		    bool bOverlap = false;
		    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
		    {
                        double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
			if (kptOverlap > maxOverlap)
			{
                            bOverlap = true;
			    if (newKeyPoint.response > (*it).response)
			    {
			        //if overlap is > t and response is higher for new kpt
				*it = newKeyPoint; //replace old key point with new one
				break;
			    }
			}
		    }
		    if (!bOverlap)
		    {
		        //only add new key point if no overlap has been found in previous NMS
			keypoints.push_back(newKeyPoint); //store new keypoint in dynamic list
		    }
		}
	    } //eof loop over cols
	} //eof loop over rows
    } else {
	cv::Ptr<cv::FeatureDetector> detector;
	if (detectorType == "FAST") {
            int threshold = 30; //difference between intensity of the central pixel and pixels of a circle around this pixel
	    bool bNMS = true; // Perform non-maxima suppression on keypoints
	    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
            detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        } else if (detectorType == "BRISK") {
	    detector = cv::BRISK::create();
        } else if (detectorType == "ORB") {
	    detector = cv::ORB::create();
        } else if (detectorType == "AKAZE") {
	    detector = cv::AKAZE::create();
        } else if (detectorType == "SIFT") {
	    detector = cv::xfeatures2d::SIFT::create();
        } else {
           throw invalid_argument("invalid key point detector");
	}
        t = (double)cv::getTickCount();
        detector->detect(img, keypoints);	
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() 
	 << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, 
			  cv::Scalar::all(-1), 
			  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}
