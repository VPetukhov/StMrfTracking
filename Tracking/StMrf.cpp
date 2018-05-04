#include "StMrf.h"
#include "Utils.h"
#include "GcWrappers.h"

#include <numeric>
#include <vector>

using namespace cv;

namespace Tracking
{
	BlockArray::id_t update_slit_objects(BlockArray &blocks, const BlockArray::Slit &slit, const Mat &frame,
	                                     const Mat &background, BlockArray::id_t new_block_id, double threshold)
	{
		const int d_xs[] = {0, 1, 1, 1, 0, -1, -1, -1, 0};
		const int d_ys[] = {-1, -1, 0, 1, 1, 1, 0, -1, 0};
		const int nds = sizeof(d_xs) / sizeof(d_xs[0]);

		std::vector<BlockArray::id_t> object_ids(slit.block_xs().size(), 0);
		for (size_t slit_block_id = 0; slit_block_id < slit.block_xs().size(); ++slit_block_id)
		{
			auto block_x = slit.block_xs()[slit_block_id];
			auto &block = blocks.at(slit.block_y(), block_x);
			if (!is_foreground(block, frame, background, threshold))
				continue;

			if (block.object_id > 0)
			{
				object_ids.at(slit_block_id) = block.object_id;
				continue;
			}

			BlockArray::id_t next_id = 0;
			for (int di = 0; di < nds; ++di)
			{
				int new_x = block_x + d_xs[di], new_y = slit.block_y() + d_ys[di];
				if (!blocks.valid_coords(new_y, new_x))
					continue;

				next_id = block.object_id;
				if (next_id != 0)
					break;
			}

			if (next_id != 0)
			{
				object_ids[slit_block_id] = next_id;
				continue;
			}

			if (slit_block_id > 1 && object_ids[slit_block_id - 1] > 0)
			{
				object_ids[slit_block_id] = object_ids[slit_block_id - 1];
				continue;
			}

			object_ids[slit_block_id] = new_block_id;
			new_block_id++;
		}

		for (size_t slit_block_id = 0; slit_block_id < slit.block_xs().size(); ++slit_block_id)
		{
			blocks.at(slit.block_y(), slit.block_xs()[slit_block_id]).object_id = object_ids.at(slit_block_id);
		}

		return *std::max(object_ids.begin(), object_ids.end()) + 1;
	}

	bool is_foreground(const BlockArray::Block &block, const Mat &frame, const Mat &background, double threshold)
	{
		auto r = cv::mean(frame(block.y_coords(), block.x_coords()) - background(block.y_coords(), block.x_coords()));
		return average(r) > threshold;
	}

	bool is_foreground(const BlockArray::Block &block, const Mat &foreground)
	{
		return (cv::mean(foreground(block.y_coords(), block.x_coords())).val[0] / 255.0) > 0.5;
	}

	group_coords_t find_group_coordinates(const cv::Mat &labels)
	{
		using id_t = BlockArray::id_t;
		double max_val;
		cv::minMaxLoc(labels, nullptr, &max_val);

		group_coords_t group_coordinates(static_cast<size_t>(max_val) + 1);
		for (size_t row = 0; row < labels.rows; ++row)
		{
			for (size_t col = 0; col < labels.cols; ++col)
			{
				id_t lab = labels.at<id_t>(row, col);
				if (lab == 0)
					continue;

				group_coordinates.at(lab).emplace_back(col, row);
			}
		}

		return group_coordinates;
	}

	group_coords_t find_group_coordinates(const object_ids_t &object_id_map, const std::set<BlockArray::id_t> &object_ids)
	{
		std::map<BlockArray::id_t, size_t> id_id_map;
		size_t i = 0;
		for (auto id : object_ids)
		{
			id_id_map.emplace(id, i++);
		}

		group_coords_t group_coordinates(object_ids.size());
		for (size_t row = 0; row < object_id_map.size(); ++row)
		{
			for (size_t col = 0; col < object_id_map.at(row).size(); ++col)
			{
				for (auto const &id : object_id_map.at(row).at(col))
				{
					group_coordinates.at(id_id_map.at(id)).emplace_back(col, row);
				}
			}
		}

		return group_coordinates;
	}

	Point find_motion_vector(const BlockArray &blocks, const Mat &frame, const Mat &old_frame,
	                         const coordinates_t &group_coords, size_t search_rad)
	{
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
	                                 const Point &coords, int search_rad)
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

//		show_image(img_region);
//		show_image(match_template);
//		show_image(similarity_map);

		return similarity_map;
	}

	cv::Point round_motion_vector(const cv::Point &motion_vec, size_t block_width, size_t block_height)
	{
		return Point(static_cast<int>(std::round(motion_vec.x / double(block_width))),
		             static_cast<int>(std::round(motion_vec.y / double(block_height))));
	}

	object_ids_t update_object_ids(const BlockArray &blocks, const cv::Mat &block_id_map, const std::vector<cv::Point> &motion_vecs,
	                               const group_coords_t &group_coords, const cv::Mat &foreground)
	{
		const int d_xs[] = {0, 1, 1, 1, 0, -1, -1, -1};
		const int d_ys[] = {-1, -1, 0, 1, 1, 1, 0, -1};
		const int nds = sizeof(d_xs) / sizeof(d_xs[0]);

		object_ids_t res_ids(blocks.height);
		for (auto &row : res_ids)
		{
			row.resize(blocks.width);
		}

		for (size_t i = 0; i < group_coords.size(); ++i)
		{
			auto const &gc = group_coords[i];
			auto const &m_vec = motion_vecs[i];

			for (auto const &coords : gc)
			{
				auto new_coords = coords + m_vec;
				if (!blocks.valid_coords(new_coords))
					continue;

				if (!is_foreground(blocks.at(new_coords), foreground))
					continue;

				auto cur_block_id = block_id_map.at<BlockArray::id_t>(coords);
				if (cur_block_id == 0)
					throw std::runtime_error("Zero block id for a group");

				res_ids.at(new_coords.y).at(new_coords.x).insert(cur_block_id);

				for (int di = 0; di < nds; ++di)
				{
					Point cur_coords = new_coords + Point(d_xs[di], d_ys[di]);
					if (!blocks.valid_coords(cur_coords))
						continue;

					auto &cur_cell = res_ids.at(cur_coords.y).at(cur_coords.x);
					if (cur_cell.find(cur_block_id) != cur_cell.end())
						continue;

					if (!is_foreground(blocks.at(cur_coords), foreground))
						continue;

					cur_cell.insert(cur_block_id);
				}
			}
		}

		return res_ids;
	}

	void reset_map_before_slit(object_ids_t &new_map, size_t slit_block_y, BlockArray::Slit::Direction vehicle_direction,
	                           const BlockArray &blocks)
	{
		size_t start_row = 0, end_row = slit_block_y;
		if (vehicle_direction == BlockArray::Slit::DOWN)
		{
			start_row = slit_block_y + 1;
			end_row = blocks.height;
		}
		else if (vehicle_direction != BlockArray::Slit::UP)
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
		std::set<BlockArray::id_t> object_ids;
		for (auto const &row : object_id_map)
		{
			for (auto const &col : row)
			{
				object_ids.insert(col.begin(), col.end());
			}
		}

		if (object_ids.size() < 2)
			return label_map_naive(object_id_map);

		auto group_coords = find_group_coordinates(object_id_map, object_ids);
		auto data_cost = unary_penalties(blocks, std::vector<BlockArray::id_t>(object_ids.begin(), object_ids.end()),
		                                 motion_vectors, group_coords, prev_pixel_map, frame, prev_frame);

//		for (size_t label = 0; label < data_cost.cols; ++label)
//		{
//			Mat res = Mat::zeros(blocks.height, blocks.width, BlockArray::cv_id_t);
//			for (size_t row = 0; row < blocks.height; ++row)
//			{
//				for (size_t col = 0; col < blocks.width; ++col)
//				{
//					res.at<id_t>(row, col) = data_cost.at<int>(id_by_coords(row, col, blocks.width), label);
//				}
//			}
//
//			show_image(heatmap(res));
//		}

//		for (size_t row = 0; row < blocks.height; ++row)
//		{
//			for (size_t col = 0; col < blocks.width; ++col)
//			{
//				bool has_label = false;
//				for (size_t label = 0; label < data_cost.cols; ++label)
//				{
//					auto id = id_by_coords(row, col, blocks.width);
//					if (data_cost.at<id_t>(id, label) < 500000)
//					{
//						if (has_label)
//						{
//							std::cout << row << ": " << data_cost(Range(id, id + 1), Range::all()) << std::endl;
//							has_label = false;
//						}
//						has_label = true;
//					}
//				}
//			}
//		}

		auto gco = gc_optimization_8_grid_graph(blocks.height, blocks.width, data_cost.cols);
		set_data_cost(gco, data_cost);
		set_smooth_cost(gco, 1); // TODO: set to 20
		gco->expansion();

		auto labels = gco_to_label_map(gco, blocks.height, blocks.width);
		double min_val, max_val;
		cv::minMaxLoc(labels, &min_val, &max_val);
//		std::cout << min_val << " " << max_val << std::endl;

		return labels;
	}

	Mat unary_penalties(const BlockArray &blocks, const std::vector<BlockArray::id_t> &object_ids,
	                    const std::vector<Point> &motion_vectors, const group_coords_t &group_coords,
	                    const Mat &prev_pixel_map, const Mat &frame, const Mat &prev_frame, double inf_val, double mult)
	{
		Mat penalties = Mat::zeros(blocks.height * blocks.width, group_coords.size() + 1, DataType<double>::type) + inf_val;
		penalties(Range::all(), cv::Range(0, 1)) = 0;

		for (size_t group_id = 0; group_id < group_coords.size(); ++group_id)
		{
			auto const &gc = group_coords.at(group_id);
			auto const &vec = motion_vectors.at(group_id);
			auto const &obj_id = object_ids.at(group_id);

			for (auto const &coords : gc)
			{
				int block_id = blocks.index(coords.y, coords.x);
				auto const &block = blocks.at(block_id);
				auto cur_colors = frame(block.y_coords(), block.x_coords());
				auto prev_x = block.x_coords() - vec.x;
				auto prev_y = block.y_coords() - vec.y;

				if (!valid_coords(prev_y.start, prev_x.start, frame.rows, frame.cols) ||
						!valid_coords(prev_y.end, prev_x.end, frame.rows, frame.cols))
				{
					penalties.at<double>(block_id, group_id + 1) = 0; // TODO: process this case more cleverly
//					std::cout << "Zero: " << prev_x.start << " " << prev_x.end << "; " << prev_y.start << " " << prev_y.end << std::endl;
				}
				else
				{
					auto prev_colors = prev_frame(prev_y, prev_x);
					auto img_diffs = mean(abs(cur_colors - prev_colors));

//					std::cout << img_diffs << std::endl;
//					std::cout << img_diffs.rows << " " << img_diffs.cols << std::endl;

					double img_diff = average(img_diffs, 3);
					double lab_diff = mean(prev_pixel_map(prev_y, prev_x) != obj_id).val[0] / 255.0;

//					std::cout << "Penalties: " << img_diff << " " << lab_diff << "; " << img_diffs.rows * img_diffs.rows << std::endl;

					penalties.at<double>(block_id, group_id + 1) = img_diff + lab_diff;
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
