#pragma once

#include <vector>
#include <unordered_set>
#include <bits/unordered_set.h>

#include "opencv2/opencv.hpp"

#include "BlockArray.h"

namespace Tracking
{
	using coordinates_t = std::vector<cv::Point>;
	using group_coords_t = std::vector<coordinates_t>;
	using object_ids_t = std::vector<std::vector<std::unordered_set<BlockArray::id_t>>>;

	bool is_foreground(const BlockArray::Block &block, const cv::Mat &foreground);
	BlockArray::id_t update_slit_objects(BlockArray &blocks, const BlockArray::Slit &slit, const cv::Mat &foreground,
	                                     int new_block_id);

	group_coords_t find_group_coordinates(const cv::Mat &labels);
	group_coords_t find_group_coordinates(const object_ids_t &object_id_map, const std::set<BlockArray::id_t> &object_ids);

	cv::Mat motion_vector_similarity_map(const BlockArray &blocks, const cv::Mat &frame, const cv::Mat &old_frame,
	                                     const cv::Point &coords, int search_rad, bool plot=false);
	cv::Point find_motion_vector(const BlockArray &blocks, const cv::Mat &frame, const cv::Mat &old_frame,
	                             const coordinates_t &group_coords, int search_rad);
	cv::Point round_motion_vector(const cv::Point &motion_vec, size_t block_width, size_t block_height);

	object_ids_t update_object_ids(const BlockArray &blocks, const cv::Mat &block_id_map, const std::vector<cv::Point> &motion_vecs,
	                               const std::vector<coordinates_t> &group_coords, const cv::Mat &foreground, int search_rad=1);

	void reset_map_before_slit(object_ids_t &new_map, size_t slit_block_y, BlockArray::Line::Direction vehicle_direction,
	                           const BlockArray &blocks);

	cv::Mat label_map_naive(const object_ids_t &object_id_map);
	cv::Mat label_map_gco(const BlockArray &blocks, const object_ids_t &object_id_map,
	                      const std::vector<cv::Point> &motion_vectors, const cv::Mat &prev_pixel_map,
	                      const cv::Mat &frame, const cv::Mat &prev_frame);

	cv::Mat unary_penalties(const BlockArray &blocks, const std::vector<BlockArray::id_t> &object_ids,
	                        const std::vector<cv::Point> &motion_vectors, const group_coords_t &group_coords,
	                        const cv::Mat &prev_pixel_map, const cv::Mat &frame, const cv::Mat &prev_frame,
	                        double inf_val = 1e3, double mult=1e3, bool img_diff_cost=true, bool lab_diff_cost=true);
};

