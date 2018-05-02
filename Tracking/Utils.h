#pragma once

#include "opencv2/opencv.hpp"

namespace Tracking
{
	cv::Mat channel_max(const cv::Mat &rgb_image);
	cv::Mat channel_any(const cv::Mat &rgb_image);
	double average(const cv::Scalar& scalar);

	void show_image(const cv::Mat &img, int time = 300000);
};

