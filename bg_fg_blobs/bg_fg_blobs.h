/*
 *  bg_fg_blobs.h
 *  OpenCVTries1
 *
 *  Created by Roy Shilkrot on 11/21/10.
 *  Copyright 2010 MIT. All rights reserved.
 *
 */

#include <vector>

using namespace std;

#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\cvaux.h>

using namespace cv;
extern Scalar refineSegments(const Mat& img,
	const Mat& mask,
	Mat& dst,
	vector<Point>& contour,
	vector<Point>& second_contour,
	Point2i& previous);