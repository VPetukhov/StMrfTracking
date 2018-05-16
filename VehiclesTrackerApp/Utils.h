#pragma once

#include <unordered_map>
#include "opencv2/opencv.hpp"
#include "BlockArray.h"

namespace Tracking
{
	using rect_map_t = std::unordered_map<BlockArray::id_t, cv::Rect>;

	cv::Mat channel_max(const cv::Mat &rgb_image);
	cv::Mat channel_any(const cv::Mat &rgb_image);
	double average(const cv::Scalar& scalar, int max_size = 0);

	void show_image(const cv::Mat &img, const std::string &wind_name="tmp", int flags=CV_GUI_EXPANDED, int time = 0);
	cv::Mat heatmap(const cv::Mat &labels);

	bool valid_coords(long row, long col, size_t height, size_t width);

	rect_map_t bounding_boxes(const BlockArray &blocks);
	void draw_grid(cv::Mat& img, const BlockArray &blocks);
	void draw_slit(cv::Mat& img, const BlockArray &blocks, const BlockArray::Slit &slit, int thickness = 1);
	cv::Mat plot_frame(const cv::Mat &frame, const BlockArray &blocks, const BlockArray::Slit &slit, const BlockArray::Line &capture);
	void save_vehicle(const cv::Mat &img, const cv::Rect &b_box, const std::string &path, size_t img_id);
};

