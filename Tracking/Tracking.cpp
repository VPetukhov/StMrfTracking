#include "Tracking.h"
#include "Utils.h"
#include "StMrf.h"

using namespace cv;

namespace Tracking
{
	bool read_frame(VideoCapture &reader, Mat &frame, int height, int width)
	{
		if (!reader.read(frame))
			return false;

		Mat small_frame(height, width, frame.type());
		resize(frame, small_frame, small_frame.size());
//		frame = small_frame;
		small_frame.convertTo(frame, DataType<float>::type, 1 / 255.0);
		return true;
	}

	Mat estimate_background(const std::string &video_file, size_t max_n_frames, double weight, size_t refine_iter_num)
	{
		VideoCapture cap(video_file);
		if(!cap.isOpened())  // check if we succeeded
			throw std::runtime_error("Can't open video: " + video_file);

		Mat frame;
		if (!read_frame(cap, frame))
			throw std::runtime_error("Video: " + video_file + " seems to be empty");

		Mat background = Mat(frame);
		background.setTo(0);

		std::vector<Mat> frames;
		for (size_t i = 0; i < max_n_frames; ++i)
		{
			if (!read_frame(cap, frame))
				break;

			frames.push_back(frame);
		}

		for (auto const &fr : frames)
		{
			background += fr;
		}

		background = background / frames.size();

		refine_background(background, frames, weight, refine_iter_num);
		return background;
	}

	Mat subtract_background(const Mat &frame, const Mat &background, double threshold)
	{
		return channel_any(abs(frame - background) > threshold);
	}

	void refine_background(Mat &background, const std::vector<Mat> &frames, double weight, size_t max_iters)
	{
		int iter = 0;
		for (float threshold : std::vector<float>({0.2, 0.1, 0.05}))
		{
			if (++iter > max_iters)
				break;

			for (auto const &frame : frames)
			{
				update_background_weighted(background, frame, threshold, weight);
			}

			for (long i = frames.size() - 1; i >= 0; --i)
			{
				update_background_weighted(background, frames.at(i), threshold, weight);
			}
		}
	}

	void update_background_weighted(cv::Mat &background, const cv::Mat &frame, double threshold, double weight)
	{
		Mat dst;
		const Mat mask = 255 - channel_any(cv::abs(frame - background) > threshold);
		Mat mask2 = mask.clone();

		medianBlur(mask, mask2, 11);
		addWeighted(background, 1 - weight, frame, weight, 0, dst);
		dst.copyTo(background, mask2);
	}

	Mat connected_components(const Mat &labels, id_map_t &step_id_map)
	{
		if (labels.type() != BlockArray::cv_id_t)
			throw std::runtime_error("Wrong image type: " + std::to_string(labels.type()));

		Mat res, input;
		labels.convertTo(input, CV_8U);

		connectedComponents(input, res);
		res.convertTo(res, labels.type());

		double n_labels;
		minMaxLoc(res, nullptr, &n_labels);

		res += labels * (n_labels + 1);

		std::map<BlockArray::id_t, BlockArray::id_t> ordered_ids;
		for (int row = 0; row < labels.rows; ++row)
		{
			auto row_prev_labs = labels.ptr<BlockArray::id_t>(row);
			auto row_cur_labs = res.ptr<BlockArray::id_t>(row);
			for (int col = 0; col < labels.cols; ++col)
			{
				auto unordered_lab = row_cur_labs[col];
				auto prev_lab = row_prev_labs[col];
				if (unordered_lab == 0)
					continue;

				auto order_it = ordered_ids.emplace(unordered_lab, ordered_ids.size() + 1);
				auto cur_lab = order_it.first->second;
				row_cur_labs[col] = cur_lab;
				auto step_map_it = step_id_map.emplace(cur_lab, prev_lab);
				if (!step_map_it.second && step_map_it.first->second != prev_lab)
					throw std::runtime_error("Repeated ids");
			}
		}

		return res;
	}

	void reverse_st_mrf_step(BlockArray &blocks, const BlockArray::Slit &slit, const std::deque<Mat> &frames,
	                         const std::deque<Mat> &backgrounds, double foreground_threshold)
	{
		if (frames.empty())
			throw std::runtime_error("Empty frames");

		for (long i = frames.size() - 2; i >= 0; --i)
		{
			day_segmentation_step(blocks, slit, frames[i], frames[i + 1], backgrounds[i], foreground_threshold);
		}

		for (size_t i = 1; i < frames.size(); ++i)
		{
			day_segmentation_step(blocks, slit, frames[i], frames[i - 1], backgrounds[i], foreground_threshold);
		}
	}

	id_map_t day_segmentation_step(BlockArray &blocks, const BlockArray::Slit &slit, const Mat &frame,
	                               const Mat &old_frame, const Mat &background, double foreground_threshold)
	{
		auto const prev_pixel_map = blocks.pixel_object_map();
		auto const object_map = blocks.object_map();
		auto const group_coords = find_group_coordinates(object_map);

		std::vector<Point> motion_vectors;
		id_map_t id_map;
		for (auto const &coords : group_coords)
		{
			motion_vectors.push_back(find_motion_vector(blocks, frame, old_frame, coords));
		}

		Mat labels = Mat::zeros(object_map.size(), BlockArray::cv_id_t);
		if (!motion_vectors.empty())
		{
			std::vector<Point> motion_vectors_rounded;
			for (auto const &vec : motion_vectors)
			{
				motion_vectors_rounded.push_back(round_motion_vector(vec, blocks.block_width, blocks.block_height));
			}

			auto foreground = subtract_background(frame, background, foreground_threshold);
			auto possible_object_ids = update_object_ids(blocks, object_map, motion_vectors_rounded, group_coords, foreground);

			reset_map_before_slit(possible_object_ids, slit.block_y(), slit.direction(), blocks);
			labels = label_map_gco(blocks, possible_object_ids, motion_vectors, prev_pixel_map, frame, old_frame);

			labels = connected_components(labels, id_map);
		}

		double max_lab;
		minMaxLoc(labels, nullptr, &max_lab);

		blocks.set_object_ids(labels);
		update_slit_objects(blocks, slit, frame, background, static_cast<BlockArray::id_t>(max_lab) + 1, foreground_threshold);
		return id_map;
	}

	Mat edge_image(const Mat &image)
	{
		Mat input;
		if (image.channels() != 1)
		{
			cvtColor(image, input, CV_BGR2GRAY);
		}
		else
		{
			input = image;
		}


		Mat res = Mat::zeros(input.size(), input.type());

		for (int i = 0; i < 9; ++i)
		{
			int row = i / 3, col = i % 3;
			if (row == 1 && col == 1)
				continue;

			Mat sum_filter = Mat::zeros(3, 3, DataType<float>::type);
			sum_filter.at<float>(1, 1) = -1;
			sum_filter.at<float>(row, col) = 1;

			Mat sum_image;
			filter2D(input, sum_image, -1, sum_filter);
			res += abs(sum_image);
		}

		Mat max_image;
		dilate(input, max_image, Mat());

		res /= max_image * 8;

		return res;
	}

	cv::Mat interlayer_feedback(const cv::Mat &frame, const cv::Mat &labels)
	{

	}

	rect_map_t bounding_boxes(const BlockArray &blocks)
	{
		rect_map_t bounding_boxes;
		for (size_t row = 0; row < blocks.height; ++row)
		{
			for (size_t col = 0; col < blocks.width; ++col)
			{
				auto const &block = blocks.at(row, col);
				if (block.object_id == 0)
					continue;

				auto bb_it = bounding_boxes.find(block.object_id);
				if (bb_it == bounding_boxes.end())
				{
					bb_it = bounding_boxes.emplace(block.object_id, Rect(block.start_x, block.start_y, 0, 0)).first;
				}

				auto const &bb = bb_it->second;
				bb_it->second.y = std::min(bb.y, static_cast<int>(block.start_y));
				bb_it->second.x = std::min(bb.x, static_cast<int>(block.start_x));
				bb_it->second.height = std::max(bb.height, static_cast<int>(block.end_y - bb.y + 1));
				bb_it->second.width = std::max(bb.width, static_cast<int>(block.end_x - bb.x + 1));
			}
		}

		return bounding_boxes;
	}

	id_set_t register_vehicle(const rect_map_t &b_boxes, const id_set_t &vehicle_ids, const id_map_t &id_map,
	                          const BlockArray::Line &capture, BlockArray::CaptureType capture_type)
	{
		id_set_t res;
		for (auto const &rect_it : b_boxes)
		{
			auto const &b_box = rect_it.second;
			if (b_box.x + b_box.width < capture.x_left || b_box.x > capture.x_right)
				continue;

			auto id_map_iter = id_map.find(rect_it.first);
			if (id_map_iter == id_map.end()) // New vehicle in the slit
				continue;

			if (vehicle_ids.find(id_map_iter->second) == vehicle_ids.end())
				continue;

			if (capture.direction == BlockArray::Line::UP)
			{
				int border_y = (capture_type == BlockArray::CROSS) ? (b_box.y + b_box.height) : b_box.y;
				if (border_y > capture.y)
					continue;
			}
			else
			{
				int border_y = (capture_type == BlockArray::CROSS) ? b_box.y : (b_box.y + b_box.height);
				if (border_y < capture.y)
					continue;
			}

			res.insert(rect_it.first);
		}

		return res;
	}

	id_set_t active_vehicle_ids(const rect_map_t &b_boxes, const BlockArray::Line &capture, BlockArray::CaptureType capture_type)
	{
		id_set_t res;
		for (auto const &rect_it : b_boxes)
		{
			auto const &b_box = rect_it.second;
			if (capture.direction == BlockArray::Line::UP)
			{
				int border_y = (capture_type == BlockArray::CROSS) ? (b_box.y + b_box.height) : b_box.y;
				if (border_y < capture.y)
					continue;
			}
			else
			{
				int border_y = (capture_type == BlockArray::CROSS) ? b_box.y : (b_box.y + b_box.height);
				if (border_y > capture.y)
					continue;
			}

			res.insert(rect_it.first);
		}

		return res;
	}

	id_set_t register_vehicle_step(BlockArray &blocks, const BlockArray::Slit &slit, const Mat &frame,
	                               const Mat &old_frame, const Mat &background, double foreground_threshold,
	                               const BlockArray::Line &capture, BlockArray::CaptureType capture_type)
	{
		auto b_boxes_prev = bounding_boxes(blocks);
		auto vehicle_ids = active_vehicle_ids(b_boxes_prev, capture, capture_type);
		auto id_map = day_segmentation_step(blocks, slit, frame, old_frame, background, foreground_threshold);
		auto b_boxes = bounding_boxes(blocks);
		return register_vehicle(b_boxes, vehicle_ids, id_map, capture, capture_type);
	}
}