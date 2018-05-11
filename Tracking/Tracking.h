#pragma once

#include <deque>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include "opencv2/opencv.hpp"

#include "BlockArray.h"

namespace Tracking
{
	using id_map_t = std::unordered_map<BlockArray::id_t, BlockArray::id_t>;
	using id_set_t = std::set<BlockArray::id_t>;
	using rect_map_t = std::unordered_map<BlockArray::id_t, cv::Rect>;

	cv::Mat estimate_background(const std::string &video_file, size_t max_n_frames=300, double weight=0.05, size_t refine_iter_num=3);
	cv::Mat subtract_background(const cv::Mat &frame, const cv::Mat &background, double threshold);

	void refine_background(cv::Mat &background, const std::vector<cv::Mat> &frames, double weight, size_t max_iters=3);
	void update_background_weighted(cv::Mat &background, const cv::Mat &frame, double threshold, double weight);
	id_set_t reverse_st_mrf_step(BlockArray &blocks, const BlockArray::Slit &slit, const std::deque<cv::Mat> &frames,
	                             const std::deque<cv::Mat> &backgrounds, double foreground_threshold,
	                             const BlockArray::Line &capture, BlockArray::CaptureType capture_type);
	void segmentation_step(BlockArray &blocks, const BlockArray::Slit &slit, const cv::Mat &frame,
	                       const cv::Mat &old_frame, const cv::Mat &foreground);
	bool read_frame(cv::VideoCapture &reader, cv::Mat &frame, int height=480, int width=600);

	cv::Mat connected_components(const cv::Mat &labels);
	cv::Mat edge_image(const cv::Mat &image);

	bool is_night(const cv::Mat &img, double threshold_red = 0.75, double threshold_bright = 0.35);

	rect_map_t bounding_boxes(const BlockArray &blocks);
	id_set_t register_vehicle_step(BlockArray &blocks, const BlockArray::Slit &slit, const cv::Mat &frame,
	                               const cv::Mat &old_frame, const cv::Mat &background, double foreground_threshold,
	                               const BlockArray::Line &capture, BlockArray::CaptureType capture_type);
	id_set_t register_vehicle(const rect_map_t &b_boxes, const id_set_t &vehicle_ids, const BlockArray::Line &capture,
	                          BlockArray::CaptureType capture_type);
	id_set_t active_vehicle_ids(const rect_map_t &b_boxes, const BlockArray::Line &capture, BlockArray::CaptureType capture_type);
}

