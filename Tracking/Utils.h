#pragma once

#include "opencv2/opencv.hpp"

namespace Tracking
{
	cv::Mat channel_max(const cv::Mat &rgb_image);
	cv::Mat channel_any(const cv::Mat &rgb_image);
	double average(const cv::Scalar& scalar, int max_size = 0);

	void show_image(const cv::Mat &img, int time = 300000, const std::string &wind_name="tmp");
	cv::Mat heatmap(const cv::Mat &labels);

	bool valid_coords(long row, long col, size_t height, size_t width);
};

