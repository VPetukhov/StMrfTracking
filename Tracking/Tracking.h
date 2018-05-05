#pragma once

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

#include "BlockArray.h"

namespace Tracking
{
	cv::Mat estimate_background(const std::string &video_file, size_t max_n_frames=300, double weight=0.05, size_t refine_iter_num=3);
	cv::Mat subtract_background(const cv::Mat &frame, const cv::Mat &background, double threshold);

	void refine_background(cv::Mat &background, const std::vector<cv::Mat> &frames, double weight, size_t max_iters=3);
	void update_background_weighted(cv::Mat &background, const cv::Mat &frame, double threshold, double weight);
	void day_segmentation_step(BlockArray &blocks, const BlockArray::Slit &slit, const cv::Mat &frame, const cv::Mat &old_frame,
	                           cv::Mat &foreground, const cv::Mat &background, double foreground_threshold);
	bool read_frame(cv::VideoCapture &reader, cv::Mat &frame, int height=480, int width=600);

	cv::Mat connected_components(const cv::Mat &labels, std::map<BlockArray::id_t, BlockArray::id_t> &id_map);
	cv::Mat edge_image(const cv::Mat &image);
}

