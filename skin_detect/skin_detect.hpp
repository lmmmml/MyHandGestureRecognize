#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv\cv.hpp>
#include <iostream>

using namespace cv;
Mat GetSkin(Mat const &src);
