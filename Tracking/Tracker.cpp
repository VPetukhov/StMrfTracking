#include "Tracker.h"
#include "Tracking.h"
#include "NightDetection.h"
#include "StMrf.h"

using namespace cv;

namespace Tracking
{

	Tracker::Tracker(double foreground_threshold, double background_update_weight, int reverse_history_size,
	                 int search_radius, double block_foreground_threshold,
	                 double edge_threshold, double edge_brightness_threshold, double interval_threshold, int min_edge_hamming_dist,
	                 const Mat &background, const BlockArray::Slit &slit,
	                 const BlockArray::Capture &capture, const BlockArray &blocks)
		: slit(slit)
		, capture(capture)
		, foreground_threshold(foreground_threshold)
		, background_update_weight(background_update_weight)
		, reverse_history_size(reverse_history_size)
		, search_radius(search_radius)
		, block_foreground_threshold(block_foreground_threshold)
		, edge_threshold(edge_threshold)
		, edge_brightness_threshold(edge_brightness_threshold)
		, interval_threshold(interval_threshold)
		, min_edge_hamming_dist(min_edge_hamming_dist)
		, _background(background.clone())
		, _blocks(blocks)
	{}

	void Tracker::add_frame(const cv::Mat &frame)
	{
		update_background_weighted(this->_background, frame, this->foreground_threshold, this->background_update_weight);

		this->_frames.push_back(frame.clone());
		this->_backgrounds.push_back(this->_background.clone());

		if (this->_frames.size() > this->reverse_history_size)
		{
			this->_frames.pop_front();
			this->_backgrounds.pop_front();
		}
	}

	const BlockArray &Tracker::blocks() const
	{
		return this->_blocks;
	}

	BlockArray &Tracker::blocks()
	{
		return this->_blocks;
	}

	id_set_t Tracker::register_vehicle_step(const cv::Mat &frame, const cv::Mat &prev_frame, const cv::Mat &background)
	{
		auto labels = connected_components(this->_blocks.object_map());
		this->_blocks.set_object_ids(labels);

		auto b_boxes_prev = bounding_boxes(this->_blocks);
		auto vehicle_ids = active_vehicle_ids(b_boxes_prev, this->capture);

		Mat foreground;
		foreground = subtract_background(frame, background, foreground_threshold);
		if (is_night(frame))
		{
			foreground = min(foreground, detect_headlights(frame));
		}
		else
		{
			foreground = min(foreground, 255 - shadow_mask(frame, background));
		}

		auto next_label_id = this->segmentation_step(frame, prev_frame, foreground);
		this->interlayer_feedback(frame, next_label_id);
		auto b_boxes = bounding_boxes(this->_blocks);
		return register_vehicle(b_boxes, vehicle_ids, this->capture);
	}

	id_set_t Tracker::reverse_st_mrf_step()
	{
		if (this->_frames.empty())
			throw std::runtime_error("Empty frames");

		id_set_t ids;
		for (long i = this->_frames.size() - 2; i >= 0; --i)
		{
			ids = this->register_vehicle_step(this->_frames[i], this->_frames[i + 1], this->_backgrounds[i]);
		}

		for (size_t i = 1; i < this->_frames.size(); ++i)
		{
			ids = this->register_vehicle_step(this->_frames[i], this->_frames[i - 1], this->_backgrounds[i]);
		}

		return ids;
	}

	std::vector<bool> Tracker::column_edge_line(const cv::Mat &edges, size_t column_id) const
	{
		std::vector<bool> line(this->_blocks.height);
		for (size_t row_id = 0; row_id < this->_blocks.height; ++row_id)
		{
			auto &block = this->_blocks.at(row_id, column_id);
			Mat reduced_col;
			reduce(edges(block.y_coords(), block.x_coords()), reduced_col, 1, CV_REDUCE_MAX);
			double edge_frac = mean(reduced_col).val[0] / 255.0;
			line[row_id] = (edge_frac > this->edge_threshold);
		}

		return line;
	}

	Tracker::Interval Tracker::longest_distant_interval(const std::vector<bool> &cur_line,
	                                                    const std::vector<bool> &prev_line, size_t column_id) const
	{
		if (cur_line.size() != prev_line.size() || prev_line.size() != this->_blocks.height)
			throw std::runtime_error("Lines must have equal size");

		Interval cur_interval, max_interval;
		for (size_t row_id = 0; row_id < this->_blocks.height; ++row_id)
		{
			auto &block = this->_blocks.at(row_id, column_id);
			bool new_id = block.object_id != cur_interval.object_id;
			if (new_id || cur_line.at(row_id) == prev_line.at(row_id))
			{
				if (cur_interval.length > max_interval.length)
				{
					max_interval = cur_interval;
				}

				cur_interval = Interval(row_id, 0, new_id ? 0 : cur_interval.const_obj_length, block.object_id);
			}

			if (block.object_id != 0)
			{
				cur_interval.const_obj_length++;
				if (cur_line.at(row_id) != prev_line.at(row_id))
				{
					cur_interval.length++;
				}
			}
		}

		return max_interval;
	}

	void Tracker::interlayer_feedback(const cv::Mat &frame, BlockArray::id_t new_id)
	{
		std::map<int, int> hamming_per_id, id_height;
		Mat edges = edge_image(frame) > this->edge_brightness_threshold;

		std::vector<BlockArray::id_t> new_ids(this->_blocks.height, 0);
		auto prev_line = this->column_edge_line(edges, 0);
		for (size_t col_id = 1; col_id < this->_blocks.width - 1; ++col_id)
		{
			auto cur_line = this->column_edge_line(edges, col_id);
			auto interval = this->longest_distant_interval(cur_line, prev_line, col_id);
			prev_line = cur_line;

			if (interval.object_id != 0 &&
				static_cast<double>(interval.length) / interval.const_obj_length > this->interval_threshold &&
				interval.length >= this->min_edge_hamming_dist)
			{
				for (int row_id = interval.start_id; row_id < interval.start_id+ interval.length; ++row_id)
				{
					new_ids.at(row_id) = new_id;
				}

				new_id++;
			}

			for (size_t row_id = 0; row_id < this->_blocks.height; ++row_id)
			{
				auto &block = this->_blocks.at(row_id, col_id);
				auto cur_new_id = new_ids.at(row_id);
				if (cur_new_id == 0)
					continue;

				if (block.object_id != this->_blocks.at(row_id, col_id + 1).object_id)
				{
					new_ids[row_id] = 0;
				}

				block.object_id = cur_new_id;
			}
		}
	}

	BlockArray::id_t Tracker::segmentation_step(const Mat &frame, const Mat &old_frame, const Mat &foreground)
	{
		auto const prev_pixel_map = this->_blocks.pixel_object_map();
		auto const object_map = this->_blocks.object_map();
		auto const group_coords = find_group_coordinates(object_map);

		std::vector<Point> motion_vectors;
		for (auto const &coords : group_coords)
		{
			auto mv = find_motion_vector(this->_blocks, frame, old_frame, coords, this->search_radius);
			motion_vectors.push_back(mv);
		}

		Mat labels = Mat::zeros(object_map.size(), BlockArray::cv_id_t);
		if (!motion_vectors.empty())
		{
			std::vector<Point> motion_vectors_rounded;
			for (auto const &vec : motion_vectors)
			{
				motion_vectors_rounded.push_back(round_motion_vector(vec, this->_blocks.block_width, this->_blocks.block_height));
			}

			auto possible_object_ids = this->update_object_ids(object_map, motion_vectors_rounded, group_coords, foreground);

			reset_map_before_slit(possible_object_ids, this->slit.block_y(), this->slit.direction(), this->_blocks);
			labels = label_map_gco(this->_blocks, possible_object_ids, motion_vectors, prev_pixel_map, frame, old_frame);
		}

		double max_lab;
		minMaxLoc(labels, nullptr, &max_lab);

		this->_blocks.set_object_ids(labels);
		return this->update_slit_objects(foreground, static_cast<BlockArray::id_t>(max_lab) + 1);
	}

	object_ids_t Tracker::update_object_ids(const cv::Mat &block_id_map, const std::vector<cv::Point> &motion_vecs,
	                                        const group_coords_t &group_coords, const cv::Mat &foreground) const
	{
		object_ids_t res_ids(this->_blocks.height);
		for (auto &row : res_ids)
		{
			row.resize(this->_blocks.width);
		}

		for (size_t i = 0; i < group_coords.size(); ++i)
		{
			auto const &gc = group_coords[i];
			auto const &m_vec = motion_vecs[i];

			for (auto const &coords : gc)
			{
				auto new_coords = coords + m_vec;
				if (!this->_blocks.valid_coords(new_coords))
					continue;

				if (!is_foreground(this->_blocks.at(new_coords), foreground, this->block_foreground_threshold))
					continue;

				auto cur_block_id = block_id_map.at<BlockArray::id_t>(coords);
				if (cur_block_id == 0)
					throw std::runtime_error("Zero block id for a group");

				res_ids.at(new_coords.y).at(new_coords.x).insert(cur_block_id);

				for (int new_y = new_coords.y - this->search_radius; new_y <= new_coords.y + this->search_radius; ++new_y)
				{
					for (int new_x = new_coords.x - this->search_radius; new_x <= new_coords.x + this->search_radius; ++new_x)
					{
						Point cur_coords(new_x, new_y);
						if (!this->_blocks.valid_coords(cur_coords))
							continue;

						auto &cur_cell = res_ids.at(cur_coords.y).at(cur_coords.x);
						if (cur_cell.find(cur_block_id) != cur_cell.end())
							continue;

						if (!is_foreground(this->_blocks.at(cur_coords), foreground, this->block_foreground_threshold))
							continue;

						cur_cell.insert(cur_block_id);
					}
				}
			}
		}

		return res_ids;
	}

	BlockArray::id_t Tracker::update_slit_objects(const cv::Mat &foreground, BlockArray::id_t new_block_id)
	{
		const int d_xs[] = {0, 1, 1, 1, 0, -1, -1, -1, 0};
		const int d_ys[] = {-1, -1, 0, 1, 1, 1, 0, -1, 0};
		const int nds = sizeof(d_xs) / sizeof(d_xs[0]);

		std::vector<BlockArray::id_t> object_ids(slit.block_xs().size(), 0);
		for (size_t slit_block_id = 0; slit_block_id < slit.block_xs().size(); ++slit_block_id)
		{
			auto block_x = slit.block_xs()[slit_block_id];
			auto &block = this->_blocks.at(slit.block_y(), block_x);
			if (!is_foreground(block, foreground, this->block_foreground_threshold))
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
				if (!this->_blocks.valid_coords(new_y, new_x))
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
			this->_blocks.at(slit.block_y(), slit.block_xs()[slit_block_id]).object_id = object_ids.at(slit_block_id);
		}

		return *std::max(object_ids.begin(), object_ids.end()) + 1;
	}
}