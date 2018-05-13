#include "Tracking.h"
#include "Utils.h"
#include "StMrf.h"
#include "NightDetection.h"

using namespace cv;

namespace Tracking
{
	bool read_frame(VideoCapture &reader, Mat &frame, int height, int width)
	{
		if (!reader.read(frame))
			return false;

		Mat small_frame(height, width, frame.type());
		resize(frame, small_frame, small_frame.size());
		small_frame.convertTo(frame, DataType<float>::type, 1 / 255.0);
		return true;
	}

	Mat estimate_background(const std::string &video_file, size_t max_n_frames, double weight, size_t refine_iter_num)
	{
		VideoCapture cap(video_file);
		if (!cap.isOpened())
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

	Mat connected_components(const Mat &labels)
	{
		if (labels.channels() != 1)
			throw std::runtime_error("Wrong number of channels: " + std::to_string(labels.channels()));

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
			auto row_cur_labs = res.ptr<BlockArray::id_t>(row);
			for (int col = 0; col < labels.cols; ++col)
			{
				auto unordered_lab = row_cur_labs[col];
				if (unordered_lab == 0)
					continue;

				auto order_it = ordered_ids.emplace(unordered_lab, ordered_ids.size() + 1);
				auto cur_lab = order_it.first->second;
				row_cur_labs[col] = cur_lab;
			}
		}

		return res;
	}

	Mat edge_image(const Mat &image)
	{
		Mat input;
		if (image.channels() != 1)
		{
			cvtColor(image, input, CV_RGB2GRAY);
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
				bb_it->second.height = std::max(bb.height, static_cast<int>(block.end_y - bb.y));
				bb_it->second.width = std::max(bb.width, static_cast<int>(block.end_x - bb.x));
			}
		}

		return bounding_boxes;
	}

	id_set_t register_vehicle(const rect_map_t &b_boxes, const id_set_t &vehicle_ids, const BlockArray::Capture &capture)
	{
		id_set_t res;
		for (auto const &rect_it : b_boxes)
		{
			auto const &b_box = rect_it.second;
			if (b_box.x + b_box.width < capture.x_left || b_box.x > capture.x_right)
				continue;

			if (vehicle_ids.find(rect_it.first) == vehicle_ids.end())
				continue;

			if (capture.direction == BlockArray::Line::UP)
			{
				int border_y = (capture.type == BlockArray::CROSS) ? (b_box.y + b_box.height) : b_box.y;
				if (border_y > capture.y)
					continue;
			}
			else
			{
				int border_y = (capture.type == BlockArray::CROSS) ? b_box.y : (b_box.y + b_box.height);
				if (border_y < capture.y)
					continue;
			}

			res.insert(rect_it.first);
		}

		return res;
	}

	id_set_t
	active_vehicle_ids(const rect_map_t &b_boxes, const BlockArray::Capture &capture)
	{
		id_set_t res;
		for (auto const &rect_it : b_boxes)
		{
			auto const &b_box = rect_it.second;
			if (capture.direction == BlockArray::Line::UP)
			{
				int border_y = (capture.type == BlockArray::CROSS) ? (b_box.y + b_box.height) : b_box.y;
				if (border_y < capture.y)
					continue;
			}
			else
			{
				int border_y = (capture.type == BlockArray::CROSS) ? b_box.y : (b_box.y + b_box.height);
				if (border_y > capture.y)
					continue;
			}

			res.insert(rect_it.first);
		}

		return res;
	}

	void hsv_channels(const Mat &img, Mat *hsv)
	{
		Mat inp = img.clone();
		if (img.type() == CV_32FC3 || img.type() == CV_64FC3)
		{
			inp = inp * 255;
			inp.convertTo(inp, CV_8UC1);
		}

		Mat hsv_img;
		cvtColor(inp, hsv_img, COLOR_RGB2HSV);
		split(hsv_img, hsv);
	}

	bool is_night(const Mat &img, double threshold_red, double threshold_bright)
	{
		Mat hsv[3];
		hsv_channels(img, hsv);

		double red_frac = mean(hsv[0] < 0.2 * 255 | hsv[0] > 0.8 * 255).val[0] / 255.0;
		double bright_frac = mean(hsv[2] > 150).val[0] / 255.0;

		return (red_frac > threshold_red) && (bright_frac < threshold_bright);
	}

	Mat shadow_mask(const Mat &frame, const Mat &background, double min_ratio, double max_ratio, double min_s, double min_h)
	{
		Mat hsv_frame[3], hsv_background[3];
		hsv_channels(frame, hsv_frame);
		hsv_channels(background, hsv_background);
		hsv_frame[2].convertTo(hsv_frame[2], DataType<double>::type);
		hsv_background[2].convertTo(hsv_background[2], DataType<double>::type);

		Mat h_mask = abs(hsv_frame[0] - hsv_background[0]) >= (min_h * 255);
		Mat s_mask = (hsv_frame[1] - hsv_background[1]) >= (min_s * 255);
		Mat v_ratio = hsv_frame[2] / hsv_background[2];
		Mat v_mask = (min_ratio <= v_ratio) & (v_ratio <= max_ratio);

		return h_mask & s_mask & v_mask;
	}
}