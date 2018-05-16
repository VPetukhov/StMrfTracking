#include "Utils.h"

using namespace cv;

namespace Tracking
{
	Mat channel_max(const Mat &rgb_image)
	{
		Mat temp, max_channel, bgr[3];

		split(rgb_image, bgr);
		cv::max(bgr[0], bgr[1], temp);
		cv::max(temp, bgr[2], max_channel);

		max_channel.convertTo(temp, DataType<float>::type, 1 / 255.0);

		return temp;
	}

	Mat channel_any(const Mat &rgb_image)
	{
		return (channel_max(rgb_image) > 1e-10);
	}

	double average(const cv::Scalar& scalar, int max_size)
	{
		double res = 0;
		int n_vals = scalar.rows * scalar.cols;
		if (max_size > 0)
		{
			n_vals = std::min(max_size, n_vals);
		}

		for (size_t i = 0; i < n_vals; ++i)
		{
			res += scalar.val[i];
		}

		return res / n_vals;
	}

	void show_image(const cv::Mat &img, const std::string &wind_name, int flags, int time)
	{
		namedWindow(wind_name, flags);
		imshow(wind_name, img);

		if(waitKey(time) >= 0)
			return;
	}

	Mat heatmap(const Mat &labels)
	{
		Mat res = Mat::zeros(labels.rows, labels.cols, CV_8UC3);
		for (size_t row = 0; row < res.rows; ++row)
		{
			for (size_t col = 0; col < res.cols; ++col)
			{
				if (labels.at<int>(row, col) == 0)
					continue;

				srand(labels.at<int>(row, col));
				res.at<Vec3b>(row, col, 0) = Vec3b(rand() % 255, rand() % 255, rand() % 255);
			}
		}
		return res;
	}

	bool valid_coords(long row, long col, size_t height, size_t width)
	{
		return (col >= 0) && (row >= 0) && (col < width) && (row < height);
	}

	void draw_grid(Mat& img, const BlockArray &blocks)
	{
		for (size_t i = 0; i < blocks.height * blocks.width; ++i)
		{
			auto const &block = blocks.at(i);
			rectangle(img, Point(block.start_x, block.start_y), Point(block.end_x, block.end_y), CV_RGB(0, 255, 0));
		}
	}

	void draw_slit(Mat& img, const BlockArray &blocks, const BlockArray::Slit &slit, int thickness)
	{
		auto block_start = blocks.at(slit.block_y(), slit.block_xs().front());
		auto block_end = blocks.at(slit.block_y(), slit.block_xs().back());

		line(img, Point(block_start.start_x, block_start.start_y), Point(block_end.end_x, block_start.start_y), CV_RGB(1, 0, 0), thickness);
		line(img, Point(block_start.start_x, block_start.end_y), Point(block_end.end_x, block_start.end_y), CV_RGB(1, 0, 0), thickness);
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

	Mat plot_frame(const Mat &frame, const BlockArray &blocks, const BlockArray::Slit &slit, const BlockArray::Line &capture)
	{
		Mat plot_img = frame.clone() * 255;
		plot_img.convertTo(plot_img, CV_8UC3);

		for (auto const &bb : bounding_boxes(blocks))
		{
			rectangle(plot_img, bb.second, CV_RGB(0, 255, 0), 1);
		}
		draw_slit(plot_img, blocks, slit, 2);
		line(plot_img, Point(capture.x_left, capture.y), Point(capture.x_right, capture.y), CV_RGB(0, 0, 255), 2);

		return plot_img;
	}

	void save_vehicle(const Mat &img, const Rect &b_box, const std::string &path, size_t img_id)
	{
		Mat out_img = Mat(img, b_box) * 255;
		out_img.convertTo(out_img, CV_8UC3);

		auto out_filename = path + "\\v" + std::to_string(img_id) + ".png";
		if (!imwrite(out_filename, out_img))
			throw std::runtime_error("Can't write image: '" + out_filename + "'");
	}
}
