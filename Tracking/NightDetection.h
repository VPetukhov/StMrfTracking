#pragma once

#include "opencv2/opencv.hpp"

namespace Tracking
{
	cv::Mat log_filter(const cv::Mat &img, double response_threshold = 100. / 255., int monochrome_threshold = 200);
	cv::Mat detect_headlights(const cv::Mat &img, double scale_factor = 2);
}

