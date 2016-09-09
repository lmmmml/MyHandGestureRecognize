#include "..\skin_detect\skin_detect.hpp"
#include <opencv2\opencv.hpp>
#include <opencv\highgui.h>
#include <opencv2\imgproc.hpp>
#include "..\vibe\vibe-background-sequential.h"
#include <math.h>
#include <string>
#include <sstream>
using namespace cv;
using namespace std;

void processVideo(char* videoFilename);
void detect(Mat const & src, vector<vector<Point>> contours);
int main(int argc, char* argv[])
{
	namedWindow("Frame");
	namedWindow("Segmentation by ViBe");

	namedWindow("Hand Skin");

	processVideo("./M2U00253.MPG");
	//processVideo(0);
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

		GaussianBlur(frame, frame, Size(3, 3), 21);
		//medianBlur(frame, frame, 3);
		frame.copyTo(dst);
		skin = GetSkin(frame);
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(skin, skin, element);
		morphologyEx(skin, skin, MORPH_OPEN, element);
		dilate(skin, skin, element);
		morphologyEx(skin, skin, MORPH_CLOSE, element);
		medianBlur(frame, frame, 3);
		imshow("Frame", skin);
		cvtColor(skin, frame, CV_BGR2YCrCb);


		if (frameNumber == 1)
		{
			segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
			model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
			libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);
		}
		if (frameNumber > 20)
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
			findContours(skin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
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
	vector<Point> biggestContour;
	int c = 0;
	RotatedRect box;
	Point2f points[4];
	vector<int> FilteredHulls;
	vector<pair<Point, double> > palm_centers;
	int fingerNum = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		area = contourArea(contours[i]);
		if (area > areamax) {
			areamax = area;
			biggestContour = contours[i];
		}
	}
	if (areamax >= 8000)
	{
		vector<vector<Point> > tcontours;
		tcontours.push_back(biggestContour);
		drawContours(src, tcontours, -1, cv::Scalar(0, 0, 255), 2);

		approxPolyDP(Mat(tcontours[0]), tcontours[0], 1, true);
		biggestContour = tcontours[0];
		vector<vector<Point> > hulls(1);
		vector<vector<int> > hullsI(1);
		convexHull(Mat(tcontours[0]), hulls[0], false);
		convexHull(Mat(tcontours[0]), hullsI[0], false);

		drawContours(src, hulls, -1, cv::Scalar(0, 255, 0), 2);

		box = minAreaRect(Mat(biggestContour));

		box.points(points);

		vector<Point2f> ps(4);
		for (int i = 0; i < 4; i++)
		{
			ps[i] = Point2f(points[i].x, points[i].y);
			line(src, points[i], points[(i + 1) % 4], Scalar(255, 0, 0), 2);
		}

		circle(src, box.center, 3, Scalar(200, 125, 75));


		for (int i = 0; i < hulls[0].size(); i++)
		{
			if (sqrtf(powf(hulls[0][i].x - hulls[0][i + 1].x, 2) + powf(hulls[0][i].y + hulls[0][i + 1].y, 2)) > box.size.width / 2)
			{
				FilteredHulls.push_back(hullsI[0][i]);
			}
		}
		vector<Vec4i> detects;
		convexityDefects(biggestContour, FilteredHulls, detects);
		
		if (detects.size()>=3)
		{
			for (int i = 0; i < detects.size(); i++)
			{
				Point startPoint(biggestContour[detects[i][0]]);
				Point endPoint(biggestContour[detects[i][1]]);
				Point depthPoint(biggestContour[detects[i][2]]);
				circle(src, startPoint, 5, Scalar(200, 125, 75),5);
				circle(src, depthPoint, 5, Scalar(200, 0, 25), 5);
				circle(src, endPoint, 5, Scalar(0, 255, 25), 5);
				if ((startPoint.y<box.center.y||depthPoint.y<box.center.y)&&(startPoint.y<depthPoint.y)&&(sqrtf(powf(startPoint.x-depthPoint.x,2)+pow(startPoint.y-depthPoint.y,2))>box.size.height/6.5)&&(sqrtf(powf(startPoint.x - depthPoint.x, 2) + pow(startPoint.y - depthPoint.y, 2))<box.size.height*0.75))
				{
					++fingerNum;
					line(src, startPoint, depthPoint, Scalar(0, 255, 255), 4);
				}
			}
			
		}
		stringstream ss;
		ss << fingerNum;
		putText(src, ss.str(), Point(200, 100), FONT_HERSHEY_SIMPLEX,3,Scalar(255,128,128),5);
		//drawContours(src, FilteredHulls, -1, cv::Scalar(0, 255, 0), 2);
		/*convexityDefects(tcontours,hulls,)*/
	}
}
