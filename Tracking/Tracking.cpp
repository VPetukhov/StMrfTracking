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

	Mat connected_components(const Mat &labels, std::map<BlockArray::id_t, BlockArray::id_t> &id_map)
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

		std::map<BlockArray::id_t, BlockArray::id_t> new_ids;
		for (int row = 0; row < labels.rows; ++row)
		{
			auto row_lab_data = labels.ptr<BlockArray::id_t>(row);
			auto row_res_data = res.ptr<BlockArray::id_t>(row);
			for (int col = 0; col < labels.cols; ++col)
			{
				auto cur_lab = row_res_data[col];
				auto prev_lab = row_lab_data[col];
				if (cur_lab == 0)
					continue;

				auto iter = new_ids.emplace(cur_lab, new_ids.size() + 1);
				row_res_data[col] = iter.first->second;
				auto it2 = id_map.emplace(iter.first->second, prev_lab);
				if (!it2.second && it2.first->second != prev_lab)
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

	void day_segmentation_step(BlockArray &blocks, const BlockArray::Slit &slit, const Mat &frame, const Mat &old_frame,
	                           const Mat &background, double foreground_threshold)
	{
		auto const prev_pixel_map = blocks.pixel_object_map();
		auto const object_map = blocks.object_map();
		auto const group_coords = find_group_coordinates(object_map);

		std::vector<Point> motion_vectors;
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

			std::map<BlockArray::id_t, BlockArray::id_t> id_map;
			labels = connected_components(labels, id_map);
		}

		double max_lab;
		minMaxLoc(labels, nullptr, &max_lab);

		blocks.set_object_ids(labels);
		update_slit_objects(blocks, slit, frame, background, static_cast<BlockArray::id_t>(max_lab) + 1, foreground_threshold);
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
}