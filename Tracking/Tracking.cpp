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

			std::cout << threshold << std::endl;
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

	void day_segmentation_step(BlockArray &blocks, const BlockArray::Slit &slit, const Mat &frame, const Mat &old_frame,
	                           Mat &foreground, const Mat &background, double foreground_threshold)
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

			foreground = subtract_background(frame, background, foreground_threshold);
			auto possible_object_ids = update_object_ids(blocks, object_map, motion_vectors_rounded, group_coords, foreground);

			reset_map_before_slit(possible_object_ids, slit.block_y(), slit.direction(), blocks);
			labels = label_map_gco(blocks, possible_object_ids, motion_vectors, prev_pixel_map, frame, old_frame);
			labels.convertTo(labels, CV_8U);
			connectedComponents(labels, labels);
		}

		double max_lab;
		minMaxLoc(labels, nullptr, &max_lab);

		blocks.set_object_ids(labels);
		update_slit_objects(blocks, slit, frame, background, static_cast<BlockArray::id_t>(max_lab) + 1, foreground_threshold);
	}

	Mat edge_image_hand_based(const Mat &image, double alpha=80, double beta = 0.02)
	{
		using pixel_t = unsigned char;
		Mat input;
		if (image.channels() != 1)
		{
			cvtColor(image, input, CV_BGR2GRAY);
		}
		else
		{
			input = image;
		}

		input.convertTo(input, DataType<pixel_t>::type);

		Mat res(input.size(), input.type());
		for (int row = 0; row < input.rows; ++row)
		{
			int row_min = std::max(0, row - 1);
			int row_max = std::min(input.rows - 1, row + 1);

			auto res_row_data = res.ptr<pixel_t>(row);
			auto in_row_data = input.ptr<pixel_t>(row);
			for (int col = 0; col < input.cols; ++col)
			{
				int col_min = max(0, col - 1);
				int col_max = min(input.cols - 1, col + 1);

				auto cur_pixel = in_row_data[col];

				pixel_t max_val = 0, sum = 0;
				for (int sub_row = row_min; sub_row <= row_max; ++sub_row)
				{
					auto sub_row_data = input.ptr<pixel_t>(sub_row);
					for (int sub_col = col_min; sub_col <= col_max; ++sub_col)
					{
						auto adj_pixel = sub_row_data[sub_col];
						sum += std::abs(adj_pixel - cur_pixel);
						max_val = std::max(max_val, adj_pixel);
					}
				}

				double u = sum / (max_val / 255.0);

				res_row_data[col] = static_cast<pixel_t>(255.0 / (1.0 + std::exp(-beta * (u - alpha))));
			}
		}

		return res;
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
}