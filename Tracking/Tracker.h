#pragma once

#include <deque>
#include "opencv2/opencv.hpp"
#include "BlockArray.h"
#include "StMrf.h"
#include "Tracking.h"

namespace Tracking
{
	class Tracker
	{
	public:
		const BlockArray::Slit slit;
		const BlockArray::Capture capture;

	private:
		const double foreground_threshold;
		const double background_update_weight;
		const int reverse_history_size;
		const int search_radius;
		const double block_foreground_threshold;

		cv::Mat _background;
		std::deque<cv::Mat> _frames, _backgrounds;
		BlockArray _blocks;

	public:
		Tracker(double foreground_threshold, double background_update_weight, int reverse_history_size, int search_radius,
		        double block_foreground_threshold, const cv::Mat &background, const BlockArray::Slit &slit,
		        const BlockArray::Capture &capture, const BlockArray &blocks);
		void add_frame(const cv::Mat &frame);

		id_set_t register_vehicle_step(const cv::Mat &frame, const cv::Mat &prev_frame, const cv::Mat &background);
		id_set_t reverse_st_mrf_step();

		const BlockArray& blocks() const;
		BlockArray& blocks();

	private:
		void segmentation_step(const cv::Mat &frame, const cv::Mat &old_frame, const cv::Mat &foreground);
		object_ids_t update_object_ids(const cv::Mat &block_id_map, const std::vector<cv::Point> &motion_vecs,
                                        const group_coords_t &group_coords, const cv::Mat &foreground) const;
		BlockArray::id_t update_slit_objects(const cv::Mat &foreground, BlockArray::id_t new_block_id);
	};
}

