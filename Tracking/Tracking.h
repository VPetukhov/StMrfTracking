#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

namespace Tracking
{
	cv::Mat estimate_background(const std::string &video_file, size_t max_n_frames=300, double weight=0.05, size_t refine_iter_num=3);
	cv::Mat subtract_background(const cv::Mat &frame, const cv::Mat &background, double threshold);

	void refine_background(cv::Mat &background, const std::vector<cv::Mat> &frames, double weight, size_t max_iters=3);
	void update_background_weighted(cv::Mat &background, const cv::Mat &frame, double threshold, double weight);
	bool read_frame(cv::VideoCapture &reader, cv::Mat &frame, int height=480, int width=600);
}

