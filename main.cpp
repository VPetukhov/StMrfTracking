#include <iostream>
#include <string>
#include <numeric>

#include "opencv2/opencv.hpp"

#include "Tracking/BlockArray.h"
#include "Tracking/StMrf.h"
#include "Tracking/Tracking.h"
#include "Tracking/Utils.h"

Mat label_map_gco(const BlockArray &blocks, const object_ids_t &object_id_map, const std::vector<Point> &motion_vectors,
                  const Mat &prev_pixel_map, const Mat &frame, const Mat &old_frame);

using namespace cv;
using namespace Tracking;

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

int main()
{
	const std::string video_file("/home/viktor/VirtualBox VMs/Shared/VehicleTracking/vehicle_videos/4K_p2.mp4");
	VideoCapture cap(video_file);
	if(!cap.isOpened())  // check if we succeeded
		return 1;

	Mat background = estimate_background(video_file, 300, 0.05, 0);

	Mat old_frame, frame;
	read_frame(cap, old_frame);
	read_frame(cap, frame);

	const size_t block_width = 24, block_height = 30;
	const size_t slit_y = 350, slit_x_start = 0, slit_x_end = 350;
	const double threshold = 0.05;
	const std::string vehicle_direction = "up";

	BlockArray blocks(background.rows / block_height, background.cols / block_width, block_height, block_width);
	draw_grid(frame, blocks);

	if (slit_y > frame.rows - frame.rows % block_height)
		throw std::logic_error("Slit y is too large: " + std::to_string(slit_y));

	BlockArray::Slit slit(slit_y, slit_x_start, slit_x_end, block_width, block_height, BlockArray::Slit::UP);

	draw_slit(frame, blocks, slit);
	show_image(frame);

	BlockArray::id_t new_object_id = 1;

	// Loop
	auto const prev_pixel_map = blocks.pixel_object_map();
	update_slit_objects(blocks, slit, frame, background, new_object_id, threshold);

	auto const object_map = blocks.object_map();
	auto const group_coords = find_group_coordinates(object_map);

	std::cout << object_map << std::endl;

	std::vector<Point> motion_vectors;
	for (auto const &coords : group_coords)
	{
		motion_vectors.push_back(find_motion_vector(blocks, frame, old_frame, coords));
	}

	Mat labels = Mat::zeros(object_map.size(), DataType<BlockArray::id_t>::type);
	if (!motion_vectors.empty())
	{
		std::vector<Point> motion_vectors_rounded;
		for (auto const& vec : motion_vectors)
		{
			motion_vectors_rounded.push_back(round_motion_vector(vec, block_width, block_height));
		}

		auto const foreground = subtract_background(frame, background, threshold);
		auto new_map = update_object_ids(blocks, object_map, motion_vectors, group_coords, foreground);
		reset_map_before_slit(new_map, block_height, slit_y, vehicle_direction, blocks);
		labels = label_map_gco(blocks, new_map, motion_vectors, prev_pixel_map, frame, old_frame);
	}

//	namedWindow("edges", 1);
//
//	Mat res;
//	resize(map, res, Size(frame.cols, frame.rows));
//	imshow("edges", res);
//	if(waitKey(300000) >= 0)
//		return 0;
//	for(;;)
//	{
//		Mat frame;
//		Mat small_frame(480, 600, DataType<float>::type);
//
//		cap >> frame; // get a new frame from camera
//		resize(frame, small_frame, small_frame.size());
//		imshow("edges", small_frame);
//		if(waitKey(30) >= 0) break;
//	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}