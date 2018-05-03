#include <iostream>
#include <string>
#include <numeric>
#include <set>

#include "opencv2/opencv.hpp"

#include "Tracking/BlockArray.h"
#include "Tracking/StMrf.h"
#include "Tracking/Tracking.h"
#include "Tracking/Utils.h"

using namespace cv;
using namespace Tracking;

const size_t block_width = 8, block_height = 10;
const size_t slit_y = 350, slit_x_start = 0, slit_x_end = 350;
const double foreground_threshold = 0.05;
const std::string vehicle_direction = "up";

void draw_grid(Mat& img, const BlockArray &blocks)
{
	for (size_t i = 0; i < blocks.height * blocks.width; ++i)
	{
		auto const &block = blocks.at(i);
		rectangle(img, Point(block.start_x, block.start_y), Point(block.end_x, block.end_y), CV_RGB(0, 255, 0));
	}
}

void draw_slit(Mat& img, const BlockArray &blocks, const BlockArray::Slit &slit)
{
	auto block_start = blocks.at(slit.block_y(), slit.block_xs().front());
	auto block_end = blocks.at(slit.block_y(), slit.block_xs().back());

	line(img, Point(block_start.start_x, block_start.start_y), Point(block_end.end_x, block_start.start_y), CV_RGB(1, 0, 0));
	line(img, Point(block_start.start_x, block_start.end_y), Point(block_end.end_x, block_start.end_y), CV_RGB(1, 0, 0));
}

std::vector<Rect> bounding_boxes(const Mat &labels)
{
	std::map<int, Rect> bounding_boxes;
	for (int row = 0; row < labels.rows; ++row)
	{
		for (int col = 0; col < labels.cols; ++col)
		{
			auto cur_lab = labels.at<BlockArray::id_t>(row, col);
			if (cur_lab == 0)
				continue;

			auto bb_it = bounding_boxes.find(cur_lab);
			if (bb_it == bounding_boxes.end())
			{
				bb_it = bounding_boxes.emplace(cur_lab, Rect(labels.cols, labels.rows, 0, 0)).first;
			}

			auto const &bb = bb_it->second;
			bb_it->second.y = std::min(bb.y, row);
			bb_it->second.x = std::min(bb.x, col);
			bb_it->second.height = std::max(bb.height, row - bb.y + 1);
			bb_it->second.width = std::max(bb.width, col - bb.x + 1);
		}
	}

	std::vector<Rect> rects;
	for (auto const &r : bounding_boxes)
	{
		rects.push_back(r.second);
	}

	return rects;
}

int main()
{
	const std::string video_file("/home/viktor/VirtualBox VMs/Shared/VehicleTracking/vehicle_videos/4K_p2.mp4");
	VideoCapture cap(video_file);
	if(!cap.isOpened())  // check if we succeeded
		return 1;

//	Mat background = estimate_background(video_file, 300, 0.05, 3);
//
//	Mat back_out;
//	background.convertTo(back_out, DataType<int>::type, 255);
//	imwrite("./bacgkround.jpg", back_out);
	Mat back_in = imread("./bacgkround.jpg");
	Mat background;
	back_in.convertTo(background, DataType<float>::type, 1 / 255.0);

	Mat frame;
	read_frame(cap, frame);

	BlockArray blocks(background.rows / block_height, background.cols / block_width, block_height, block_width);
//	draw_grid(frame, blocks);

	if (slit_y > frame.rows - frame.rows % block_height)
		throw std::logic_error("Slit y is too large: " + std::to_string(slit_y));

	BlockArray::Slit slit(slit_y, slit_x_start, slit_x_end, block_width, block_height, BlockArray::Slit::UP);

//	draw_slit(frame, blocks, slit);
//	show_image(frame);

	Mat foreground = Mat::zeros(frame.rows, frame.cols, DataType<int>::type);
	BlockArray::id_t new_object_id = 1;

	namedWindow("edges", 1);
	// Loop
	int i = 0;
	while (true)
	{
		i++;
		auto const prev_pixel_map = blocks.pixel_object_map();
		update_slit_objects(blocks, slit, frame, background, new_object_id, foreground_threshold);

		Mat old_frame = frame;
		if (!read_frame(cap, frame))
			break;

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
				motion_vectors_rounded.push_back(round_motion_vector(vec, block_width, block_height));
			}

			foreground = subtract_background(frame, background, foreground_threshold);
			auto possible_object_ids = update_object_ids(blocks, object_map, motion_vectors_rounded, group_coords, foreground);

			reset_map_before_slit(possible_object_ids, block_height, slit_y, vehicle_direction, blocks);
			labels = label_map_gco(blocks, possible_object_ids, motion_vectors, prev_pixel_map, frame, old_frame);
		}

		double max_lab;
		minMaxLoc(labels, nullptr, &max_lab);
		new_object_id = static_cast<BlockArray::id_t>(max_lab) + 1;
		blocks.set_object_ids(labels);

		auto plot_map = blocks.pixel_object_map();
		Mat plot_img = frame.clone();

		for (auto const &bb : bounding_boxes(plot_map))
		{
			rectangle(plot_img, bb, CV_RGB(0, 255, 0), 1);
		}
		draw_slit(plot_img, blocks, slit);

		imshow("edges", plot_img);
		if(waitKey(30) >= 0)
			break;

		std::cout << ".";
	}

	return 0;
}