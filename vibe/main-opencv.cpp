/**
 * @file main-opencv.cpp
 * @date July 2014 
 * @brief An exemplative main file for the use of ViBe and OpenCV
 */
#include <iostream>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/imgproc.hpp"
#include "vibe-background-sequential.h"

using namespace cv;
using namespace std;

/** Function Headers */
void processVideo(char* videoFilename);
void detect(Mat const * src, vector<vector<Point>> contours);
/**
 * Displays instructions on how to use this program.
 */

void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use ViBe with OpenCV                            " << endl
    << "Usage:"                                                                     << endl
    << "./main-opencv <video filename>"                                             << endl
    << "for example: ./main-opencv video.avi"                                       << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}

double dist(Point x, Point y)
{
	return (x.x - y.x)*(x.x - y.x) + (x.y - y.y)*(x.y - y.y);
}

pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3)
{
	double offset = pow(p2.x, 2) + pow(p2.y, 2);
	double bc = (pow(p1.x, 2) + pow(p1.y, 2) - offset) / 2.0;
	double cd = (offset - pow(p3.x, 2) - pow(p3.y, 2)) / 2.0;
	double det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
	double TOL = 0.0000001;
	if (abs(det) < TOL) { cout << "POINTS TOO CLOSE" << endl; return make_pair(Point(0, 0), 0); }

	double idet = 1 / det;
	double centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
	double centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
	double radius = sqrt(pow(p2.x - centerx, 2) + pow(p2.y - centery, 2));

	return make_pair(Point(centerx, centery), radius);
}

/**
 * Main program. It shows how to use the grayscale version (C1R) and the RGB version (C3R). 
 */
int main(int argc, char* argv[])
{
  /* Print help information. */
  help();

  /* Check for the input parameter correctness. */
  /*if (argc != 2) {
    cerr <<"Incorrect input" << endl;
    cerr <<"exiting..." << endl;
    return EXIT_FAILURE;
  }*/

  /* Create GUI windows. */
  namedWindow("Frame");
  namedWindow("Segmentation by ViBe");
  namedWindow("FindContours");
  processVideo(0);

  /* Destroy GUI windows. */
  destroyAllWindows();
  return EXIT_SUCCESS;
}

/**
 * Processes the video. The code of ViBe is included here. 
 *
 * @param videoFilename  The name of the input video file. 
 */
void processVideo(char* videoFilename)
{
	vector<vector<Point>> contours;
  /* Create the capture object. */
  VideoCapture capture(0);

  if (!capture.isOpened()) {
    /* Error in opening the video input. */
    cerr << "Unable to open video file: " << videoFilename << endl;
    exit(EXIT_FAILURE);
  }

  /* Variables. */
  static int frameNumber = 1; /* The current frame number */
  Mat frame;                  /* Current frame. */
  Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
  int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
  Mat frameHSV;	// hsv空间
  Mat mask(frame.rows, frame.cols, CV_8UC1);	// 2值掩膜
  /* Model for ViBe. */
  vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */

  /* Read input data. ESC or 'q' for quitting. */
  while ((char)keyboard != 'q' && (char)keyboard != 27)
  {
    /* Read the current frame. */
    if (!capture.read(frame)) {
      cerr << "Unable to read next frame." << endl;
      cerr << "Exiting..." << endl;
      exit(EXIT_FAILURE);
    }

	/*medianBlur(frame, frame, 5);*/
	cvtColor(frame, frame, CV_BGR2YCrCb);
	frame.copyTo(frameHSV);
	Mat dstTemp1(frame.rows, frame.cols, CV_8UC1);
	Mat dstTemp2(frame.rows, frame.cols, CV_8UC1);
	// 对HSV空间进行量化，得到2值图像，亮的部分为手的形状
	inRange(frameHSV, Scalar(0, 30, 30), Scalar(40, 170, 256), dstTemp1);
	inRange(frameHSV, Scalar(156, 30, 30), Scalar(180, 170, 256), dstTemp2);
	bitwise_or(dstTemp1, dstTemp2, mask);
	// 形态学操作，去除噪声，并使手的边界更加清晰
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(mask, mask, element);
	morphologyEx(mask, mask, MORPH_OPEN, element);
	dilate(mask, mask, element);
	morphologyEx(mask, mask, MORPH_CLOSE, element);
	imshow("FindContours", mask);
    if ((frameNumber % 100) == 0) { cout << "Frame number = " << frameNumber << endl; }

    /* Applying ViBe.
     * If you want to use the grayscale version of ViBe (which is much faster!):
     * (1) remplace C3R by C1R in this file.
     * (2) uncomment the next line (cvtColor).
     */
	/*cvtColor(frame, frame, COLOR_BGR2GRAY);*/
	/*equalizeHist(frame, frame);*/
    if (frameNumber == 30) {
      segmentationMap = Mat(frame.rows, frame.cols, CV_8UC1);
      model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
      libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);
    }
	if (frameNumber > 30)
	{
		/* ViBe: Segmentation and updating. */
		libvibeModel_Sequential_Segmentation_8u_C3R(model, frame.data, segmentationMap.data);
		libvibeModel_Sequential_Update_8u_C3R(model, frame.data, segmentationMap.data);

		/* Post-processes the segmentation map. This step is not compulsory.
		   Note that we strongly recommend to use post-processing filters, as they
		   always smooth the segmentation map. For example, the post-processing filter
		   used for the Change Detection dataset (see http://www.changedetection.net/ )
		   is a 5x5 median filter. */
		medianBlur(segmentationMap, segmentationMap, 5); /* 3x3 median filtering */
		/*erode(segmentationMap, segmentationMap, cv::Mat());
		dilate(segmentationMap, segmentationMap, cv::Mat());*/
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(segmentationMap, segmentationMap, element);
		morphologyEx(segmentationMap, segmentationMap, MORPH_OPEN, element);
		dilate(segmentationMap, segmentationMap, element);
		morphologyEx(segmentationMap, segmentationMap, MORPH_CLOSE, element);
		imshow("Segmentation by ViBe", segmentationMap);
		findContours(segmentationMap, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		detect(&frame, contours);
		/*int size = contours.size();
		for (int i = 0; i < size; i++)
		{
			if (contourArea(contours[i]) >= 5000)
			{
				vector<vector<Point> > tcontours;
				tcontours.push_back(contours[i]);
				drawContours(frame, tcontours, -1, cv::Scalar(0, 0, 255), 2);

				
				vector<vector<Point> > hulls(1);
				vector<vector<int> > hullsI(1);
				convexHull(Mat(tcontours[0]), hulls[0], false);
				convexHull(Mat(tcontours[0]), hullsI[0], false);
			}
		}*/
	}
    /* Shows the current frame and the segmentation map. */
    imshow("Frame", frame);

    

    ++frameNumber;

    /* Gets the input from the keyboard. */
    keyboard = waitKey(1);
  }

  /* Delete capture object. */
  capture.release();

  /* Frees the model. */
  libvibeModel_Sequential_Free(model);
}

void detect(Mat const * src, vector<vector<Point>> contours)
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
		drawContours(*src, tcontours, -1, cv::Scalar(0, 0, 255), 2);
		
		approxPolyDP(Mat(tcontours[0]), tcontours[0], 3,true);
		vector<vector<Point> > hulls(1);
		vector<vector<int> > hullsI(1);
		convexHull(Mat(tcontours[0]), hulls[0], false);
		convexHull(Mat(tcontours[0]), hullsI[0], false);

		drawContours(*src, hulls, -1, cv::Scalar(0, 255, 0), 2);

		/*convexityDefects(tcontours,hulls,)*/
	}
}

