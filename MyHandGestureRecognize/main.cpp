#include "..\skin_detect\skin_detect.hpp"
#include <opencv\cv.h>
#include <opencv2\opencv.hpp>
#include <opencv\highgui.h>
#include <opencv2\imgproc.hpp>
#include "..\vibe\vibe-background-sequential.h"

using namespace cv;
using namespace std;

void processVideo(char* videoFilename);
void detect(Mat const & src, vector<vector<Point>> contours);
int main(int argc, char* argv[])
{
	namedWindow("Frame");
	namedWindow("Segmentation by ViBe");

	namedWindow("Hand Skin");

	processVideo(0);
	destroyAllWindows();
	return EXIT_SUCCESS;
}

void processVideo(char * videoFilename)
{
	vector<vector<Point>> contours;

	VideoCapture * CapPtr = NULL;
	if (videoFilename == NULL)
	{
		CapPtr = new VideoCapture(0);
	}
	else
	{
		CapPtr = new VideoCapture(videoFilename);
	}

	if (!CapPtr->isOpened())
	{
		exit(EXIT_FAILURE);
	}

	static int frameNumber = 1;
	Mat frame;
	Mat segmentationMap;
	int keyboard = 0;
	Mat skin;
	Mat dst;
	
	Mat mask(frame.rows, frame.cols, CV_8UC1);
	vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

	while ((char)keyboard != 'q' && (char)keyboard != 27)
	{
		if (!CapPtr->read(frame))
		{
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			exit(EXIT_FAILURE);
		}
		
		medianBlur(frame, frame, 5);
		frame.copyTo(dst);
		skin = GetSkin(frame);
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(skin, skin, element);
		morphologyEx(skin, skin, MORPH_OPEN, element);
		dilate(skin, skin, element);
		morphologyEx(skin, skin, MORPH_CLOSE, element);
		/*cvtColor(skin, frame, CV_BGR2YCrCb);*/

	
		if (frameNumber == 20)
		{
			segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
			model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
			libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);
		}
		if (frameNumber > 30)
		{
			/* ViBe: Segmentation and updating. */
			libvibeModel_Sequential_Segmentation_8u_C3R(model, frame.data, segmentationMap.data);
			libvibeModel_Sequential_Update_8u_C3R(model, frame.data, segmentationMap.data);
			//medianBlur(segmentationMap, segmentationMap, 5); /* 3x3 median filtering */
			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
			erode(segmentationMap, segmentationMap, element);
			morphologyEx(segmentationMap, segmentationMap, MORPH_OPEN, element);
			dilate(segmentationMap, segmentationMap, element);
			morphologyEx(segmentationMap, segmentationMap, MORPH_CLOSE, element);
			imshow("Segmentation by ViBe", segmentationMap);
			cvtColor(skin, skin, CV_BGR2GRAY);
			findContours(segmentationMap, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
			detect(dst, contours);
			imshow("Hand Skin", dst);
		}
		++frameNumber;
		keyboard = waitKey(1);
	}
	/* Delete capture object. */
	CapPtr->release();

	/* Frees the model. */
	libvibeModel_Sequential_Free(model);
}

void detect(Mat const & src, vector<vector<Point>> contours)
{
	double area = 0, areamax = 0;
	int c = 0;
	vector<pair<Point, double> > palm_centers;
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		if (area > areamax) {
			areamax = area;
			c = i;
		}
	}
	if (areamax >= 8000)
	{
		vector<vector<Point> > tcontours;
		tcontours.push_back(contours[c]);
		drawContours(src, tcontours, -1, cv::Scalar(0, 0, 255), 2);

		approxPolyDP(Mat(tcontours[0]), tcontours[0], 3, true);
		vector<vector<Point> > hulls(1);
		vector<vector<int> > hullsI(1);
		convexHull(Mat(tcontours[0]), hulls[0], false);
		convexHull(Mat(tcontours[0]), hullsI[0], false);

		drawContours(src, hulls, -1, cv::Scalar(0, 255, 0), 2);

		/*convexityDefects(tcontours,hulls,)*/
	}
}
