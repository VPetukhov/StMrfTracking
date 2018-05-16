#pragma once

#include "opencv2/opencv.hpp"

namespace Tracking
{
	cv::Mat log_filter(const cv::Mat &img, double response_threshold );
	cv::Mat detect_headlights(const cv::Mat &img, double scale_factor = 2, double response_threshold = 100. / 255.,
	                          double monochrome_threshold = 200. / 255.);
}

