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
	private:
		struct Interval
		{
			explicit Interval(size_t start_id = 0, size_t length = 0, size_t const_obj_length = 0, BlockArray::id_t object_id = 0)
				: start_id(start_id)
				, length(length)
				, const_obj_length(const_obj_length)
				, object_id(object_id)
			{}

			size_t start_id;
			size_t length;
			size_t const_obj_length;
			BlockArray::id_t object_id;
		};

	public:
		const BlockArray::Slit slit;
		const BlockArray::Capture capture;

	private:
		const double foreground_threshold;
		const double background_update_weight;
		const int reverse_history_size;
		const int search_radius;
		const double block_foreground_threshold;

		// Interlayer feedback
		const bool do_interlayer_feedback;
		const double edge_threshold;
		const double edge_brightness_threshold;
		const double interval_threshold;
		const int min_edge_hamming_dist;

		cv::Mat _background;
		std::deque<cv::Mat> _frames, _backgrounds;
		BlockArray _blocks;

	public:
		Tracker(double foreground_threshold, double background_update_weight, int reverse_history_size, int search_radius,
		        double block_foreground_threshold, bool do_interlayer_feedback,
		        double edge_threshold, double edge_brightness_threshold, double interval_threshold, int min_edge_hamming_dist,
		        const cv::Mat &background, const BlockArray::Slit &slit,
		        const BlockArray::Capture &capture, const BlockArray &blocks);
		void add_frame(const cv::Mat &frame);

		id_set_t register_vehicle_step(const cv::Mat &frame, const cv::Mat &prev_frame, const cv::Mat &background, bool do_il_feedback=true, bool return_ids=true);
		id_set_t reverse_st_mrf_step();

		const cv::Mat& background() const;
		const BlockArray& blocks() const;
		BlockArray& blocks();

	private:
		BlockArray::id_t segmentation_step(const cv::Mat &frame, const cv::Mat &old_frame, const cv::Mat &foreground);
		object_ids_t update_object_ids(const cv::Mat &block_id_map, const std::vector<cv::Point> &motion_vecs,
                                        const group_coords_t &group_coords, const cv::Mat &foreground) const;
		BlockArray::id_t update_slit_objects(const cv::Mat &foreground, BlockArray::id_t new_block_id);

		void interlayer_feedback(const cv::Mat &frame, BlockArray::id_t new_id);

		std::vector<bool> column_edge_line(const cv::Mat &edges, size_t column_id) const;
		Interval longest_distant_interval(const std::vector<bool> &cur_line, const std::vector<bool> &prev_line, size_t column_id) const;
	};
}

