#include "StMrf.h"
#include "Utils.h"
#include "GcWrappers.h"

#include <numeric>
#include <vector>

using namespace cv;

namespace Tracking
{
	bool is_foreground(const BlockArray::Block &block, const Mat &foreground, double block_foreground_threshold)
	{
		return (cv::mean(foreground(block.y_coords(), block.x_coords())).val[0] / 255.0) > block_foreground_threshold;
	}

	group_coords_t find_group_coordinates(const cv::Mat &labels)
	{
		using id_t = BlockArray::id_t;
		double max_val;
		cv::minMaxLoc(labels, nullptr, &max_val);

		group_coords_t group_coordinates(static_cast<size_t>(max_val));
		for (size_t row = 0; row < labels.rows; ++row)
		{
			for (size_t col = 0; col < labels.cols; ++col)
			{
				id_t lab = labels.at<id_t>(row, col);
				if (lab == 0)
					continue;

				group_coordinates.at(lab - 1).emplace_back(col, row);
			}
		}

		return group_coordinates;
	}

	group_coords_t find_group_coordinates(const object_ids_t &object_id_map, const std::set<BlockArray::id_t> &object_ids)
	{
		group_coords_t group_coordinates(*std::max_element(object_ids.begin(), object_ids.end()));

		for (size_t row = 0; row < object_id_map.size(); ++row)
		{
			for (size_t col = 0; col < object_id_map.at(row).size(); ++col)
			{
				for (auto const &id : object_id_map.at(row).at(col))
				{
					group_coordinates.at(id - 1).emplace_back(col, row);
				}
			}
		}

		return group_coordinates;
	}

	Point find_motion_vector(const BlockArray &blocks, const Mat &frame, const Mat &old_frame,
	                         const coordinates_t &group_coords, int search_rad)
	{
		search_rad = 2;
		Mat similarity_map = Mat::zeros(blocks.block_height * search_rad * 2 + 1, blocks.block_width * search_rad * 2 + 1, DataType<double>::type);
		for (auto const&  coords: group_coords)
		{
			auto cur_map = motion_vector_similarity_map(blocks, frame, old_frame, coords, search_rad);
			add(similarity_map, cur_map, similarity_map, noArray(), DataType<double>::type);
		}

		Point min_pos;
		minMaxLoc(similarity_map, nullptr, nullptr, &min_pos, nullptr);
		min_pos.y -= blocks.block_height * search_rad;
		min_pos.x -= blocks.block_width * search_rad;

		return min_pos;
	}

	Mat motion_vector_similarity_map(const BlockArray &blocks, const Mat &frame, const Mat &old_frame,
	                                 const Point &coords, int search_rad, bool plot)
	{
		Mat similarity_map = Mat::zeros(blocks.block_height * search_rad * 2 + 1, blocks.block_width * search_rad * 2 + 1, DataType<double>::type);
		if (!blocks.valid_coords(coords.y - search_rad, coords.x - search_rad) ||
			!blocks.valid_coords(coords.y + search_rad, coords.x + search_rad))  // TODO: replace with zero padding
			return similarity_map;

		auto const &top_left = blocks.at(coords.y - search_rad, coords.x - search_rad);
		auto const &bottom_right = blocks.at(coords.y + search_rad, coords.x + search_rad);
		auto const x_coords = Range(top_left.start_x, bottom_right.end_x);
		auto const y_coords = Range(top_left.start_y, bottom_right.end_y);

		auto const &match_template = frame(blocks.at(coords).y_coords(), blocks.at(coords).x_coords());
		auto const &img_region = old_frame(y_coords, x_coords);

		matchTemplate(img_region, match_template, similarity_map, CV_TM_SQDIFF);

		if (plot)
		{
			show_image(img_region);
			show_image(match_template);
			show_image(similarity_map);
		}

		return similarity_map;
	}

	cv::Point round_motion_vector(const cv::Point &motion_vec, size_t block_width, size_t block_height)
	{
		return Point(static_cast<int>(std::round(motion_vec.x / double(block_width))),
		             static_cast<int>(std::round(motion_vec.y / double(block_height))));
	}

	void reset_map_before_slit(object_ids_t &new_map, size_t slit_block_y, BlockArray::Line::Direction vehicle_direction,
	                           const BlockArray &blocks)
	{
		size_t start_row = 0, end_row = slit_block_y;
		if (vehicle_direction == BlockArray::Line::UP)
		{
			start_row = slit_block_y + 1;
			end_row = blocks.height;
		}
		else if (vehicle_direction != BlockArray::Line::DOWN)
			throw std::runtime_error("Wrong vehicle direction: " + std::to_string(vehicle_direction));

		for (size_t row = start_row; row < end_row; ++row)
		{
			for (size_t col = 0; col < blocks.width; ++col)
			{
				new_map.at(row).at(col).clear();
			}
		}
	}

	Mat label_map_naive(const object_ids_t &object_id_map)
	{
		size_t n_cols = object_id_map.at(0).size();
		Mat res = Mat::zeros(object_id_map.size(), n_cols, BlockArray::cv_id_t);
		for (size_t row = 0; row < object_id_map.size(); ++row)
		{
			for (size_t col = 0; col < n_cols; ++col)
			{
				auto const &cur_ids = object_id_map.at(row).at(col);
				if (cur_ids.empty())
					continue;

				res.at<BlockArray::id_t>(row, col) = *cur_ids.begin();
			}
		}

		return res;
	}

	Mat label_map_gco(const BlockArray &blocks, const object_ids_t &object_id_map, const std::vector<Point> &motion_vectors,
	                  const Mat &prev_pixel_map, const Mat &frame, const Mat &prev_frame)
	{
		size_t max_size = 0;
		std::set<BlockArray::id_t> object_ids;
		for (auto const &row : object_id_map)
		{
			for (auto const &col : row)
			{
				object_ids.insert(col.begin(), col.end());
				max_size = std::max(max_size, col.size());
			}
		}

		if (max_size < 2)
			return label_map_naive(object_id_map);

		auto group_coords = find_group_coordinates(object_id_map, object_ids);
		auto data_cost = unary_penalties(blocks, std::vector<BlockArray::id_t>(object_ids.begin(), object_ids.end()),
		                                 motion_vectors, group_coords, prev_pixel_map, frame, prev_frame);

		auto gco = gc_optimization_8_grid_graph(blocks.width, blocks.height, data_cost.cols, blocks.object_map());
		set_data_cost(gco, data_cost);
		set_smooth_cost(gco, data_cost.cols, 20);
		gco->expansion();

		auto labels = gco_to_label_map(gco, blocks.height, blocks.width);
		double min_val, max_val;
		cv::minMaxLoc(labels, &min_val, &max_val);

		return labels;
	}

	Mat unary_penalties(const BlockArray &blocks, const std::vector<BlockArray::id_t> &object_ids,
	                    const std::vector<Point> &motion_vectors, const group_coords_t &group_coords,
	                    const Mat &prev_pixel_map, const Mat &frame, const Mat &prev_frame, double inf_val, double mult,
	                    bool img_diff_cost, bool lab_diff_cost)
	{
		Mat penalties = Mat::zeros(blocks.height * blocks.width, group_coords.size() + 1, DataType<double>::type) + inf_val;
		penalties(Range::all(), cv::Range(0, 1)) = 0;

		for (auto obj_id : object_ids)
		{
			auto const &gc = group_coords.at(obj_id - 1);
			auto const &vec = motion_vectors.at(obj_id - 1);

			for (auto const &coords : gc)
			{
				int block_id = blocks.index(coords.y, coords.x);
				auto const &block = blocks.at(block_id);
				auto cur_colors = frame(block.y_coords(), block.x_coords());
				auto prev_x = block.x_coords() + vec.x;
				auto prev_y = block.y_coords() + vec.y;

				if (!valid_coords(prev_y.start, prev_x.start, prev_pixel_map.rows, prev_pixel_map.cols) ||
						!valid_coords(prev_y.end, prev_x.end, prev_pixel_map.rows, prev_pixel_map.cols))
				{
					penalties.at<double>(block_id, obj_id) = 0;
				}
				else
				{
					auto prev_colors = prev_frame(prev_y, prev_x);
					auto img_diffs = mean(abs(cur_colors - prev_colors));

					double img_diff = img_diff_cost ? average(img_diffs, 3) : 0;
					double lab_diff = lab_diff_cost ? mean(prev_pixel_map(prev_y, prev_x) != obj_id).val[0] / 255.0 : 0;

					penalties.at<double>(block_id, obj_id) = img_diff + lab_diff;
				}

				penalties.at<double>(block_id, 0) = inf_val;
			}
		}

		Mat int_penalties;
		penalties *= mult;
		penalties.convertTo(int_penalties, DataType<int>::type);
		return int_penalties;
	}
}
