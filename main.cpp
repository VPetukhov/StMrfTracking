#include <iostream>
#include <string>
#include <deque>
#include <numeric>
#include <set>
#include <limits>
#include <getopt.h>

#include "opencv2/opencv.hpp"

#include "Tracking/BlockArray.h"
#include "Tracking/StMrf.h"
#include "Tracking/Tracking.h"
#include "Tracking/Utils.h"

using namespace cv;
using namespace Tracking;

static const std::string SCRIPT_NAME = "stmrf";
const size_t NA_VALUE = std::numeric_limits<size_t>::max();

struct Params
{
	bool cant_parse = false;
	size_t block_width = 16;
	size_t block_height = 20;
	double foreground_threshold = 0.05;
	size_t slit_y = NA_VALUE; // 350
	size_t slit_x_left = NA_VALUE; // 0
	size_t slit_x_right = NA_VALUE; // 350
	BlockArray::Slit::Direction vehicle_direction = BlockArray::Slit::DOWN;
};

static void usage()
{
	std::cerr << SCRIPT_NAME <<":\n"
	          << "SYNOPSIS\n"
	          << "\t" << SCRIPT_NAME << " [options] -y slit_y -l slit_x_left -r slit_x_right video_file\n"
	          << "OPTIONS:\n"
	          << "\t-y, --slit-y: y-coordinate of the slit\n"
	          << "\t-l, --slit-x-left: left x-coordinate of the slit\n"
	          << "\t-r, --slit-x-right: right x-coordinate of the slit\n"
	          << "\t-h, --block-height: height of each block. Default: " << Params().block_height << "\n"
	          << "\t-w, --block-width: width of each block. Default: " << Params().block_width << "\n"
	          << "\t-t, --foreground-threshold: Threshold, used to distinguish background from foreground. Default: " << Params().foreground_threshold << "\n";
}

static Params parse_cmd_params(int argc, char **argv)
{
	Params params{};

	int option_index = 0;
	int c;
	static struct option long_options[] = {
			{"slit-y",     	 required_argument, nullptr, 'y'},
			{"slit-x-left",  required_argument, nullptr, 'l'},
			{"slit-x-right", required_argument, nullptr, 'r'},
			{"block-height", required_argument, nullptr, 'h'},
			{"block-width",  required_argument,	nullptr, 'w'},
			{"foreground-threshold", required_argument, nullptr, 't'},
			{nullptr, 0, nullptr, 0}
	};
	while ((c = getopt_long(argc, argv, "y:l:r:h:w:t:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
			case 'y':
				params.slit_y = strtoul(optarg, nullptr, 10);
				break;
			case 'l' :
				params.slit_x_left = strtoul(optarg, nullptr, 10);
				break;
			case 'r' :
				params.slit_x_right = strtoul(optarg, nullptr, 10);
				break;
			case 'h' :
				params.block_height = strtoul(optarg, nullptr, 10);
				break;
			case 'w' :
				params.block_width = strtoul(optarg, nullptr, 10);
				break;
			case 't' :
				params.foreground_threshold = strtod(optarg, nullptr);
				break;
			default:
				std::cerr << SCRIPT_NAME << ": unknown arguments passed: '" << (char)c <<"'"  << std::endl;
				params.cant_parse = true;
				return params;
		}
	}

	if (optind > argc - 1)
	{
		std::cerr << SCRIPT_NAME << ": video file must be supplied" << std::endl;
		params.cant_parse = true;
	}

	if (params.slit_y == NA_VALUE)
	{
		std::cerr << SCRIPT_NAME << ": slit-y parameter must be supplied" << std::endl;
		params.cant_parse = true;
	}

	if (params.slit_x_left == NA_VALUE)
	{
		std::cerr << SCRIPT_NAME << ": slit-x-left parameter must be supplied" << std::endl;
		params.cant_parse = true;
	}

	if (params.slit_x_right == NA_VALUE)
	{
		std::cerr << SCRIPT_NAME << ": slit-x-right parameter must be supplied" << std::endl;
		params.cant_parse = true;
	}

	return params;
}

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
		auto row_data = labels.ptr<BlockArray::id_t>(row);
		for (int col = 0; col < labels.cols; ++col)
		{
			auto cur_lab = row_data[col];
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

bool plot_frame(const Mat &frame, const BlockArray &blocks, const BlockArray::Slit &slit, int delay=30)
{
	auto plot_map = blocks.pixel_object_map();
	Mat plot_img = frame.clone();

	for (auto const &bb : bounding_boxes(plot_map))
	{
		rectangle(plot_img, bb, CV_RGB(0, 255, 0), 1);
	}
	draw_slit(plot_img, blocks, slit);

	imshow("edges", plot_img);
	if(waitKey(delay) >= 0)
		return false;

//	show_image(heatmap(plot_map));
//	show_image(heatmap(blocks.object_map()));
	return true;
}

int main(int argc, char **argv)
{
	Params p = parse_cmd_params(argc, argv);
	if (p.cant_parse)
		return 1;

	const std::string video_file(argv[optind]);
	VideoCapture cap(video_file);
	if(!cap.isOpened())  // check if we succeeded
	{
		std::cerr << "Can't open video file: " << video_file << std::endl;
		return 1;
	}

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

	BlockArray blocks(background.rows / p.block_height, background.cols / p.block_width, p.block_height, p.block_width);
//	draw_grid(frame, blocks);

	if (p.slit_y > frame.rows - frame.rows % p.block_height)
		throw std::logic_error("Slit y is too large: " + std::to_string(p.slit_y));

	BlockArray::Slit slit(p.slit_y, p.slit_x_left, p.slit_x_right, p.block_width, p.block_height, p.vehicle_direction);

//	draw_slit(frame, blocks, slit);
//	show_image(frame);

	namedWindow("edges", 1);

	// Loop
	int i = 0;
	Mat old_frame;
	std::deque<Mat> frames, backgrounds;
	frames.push_back(frame.clone());
	backgrounds.push_back(background.clone());

	while (read_frame(cap, frame))
	{
		if (++i % 5 != 0)
			continue;

		frames.push_back(frame.clone());
		backgrounds.push_back(background.clone());

		if (frames.size() > 5)
		{
			frames.pop_front();
			backgrounds.pop_front();
		}

		std::cout << "Step " << i << std::endl;
//		day_segmentation_step(blocks, slit, frame, old_frame, background, p.foreground_threshold);
//		old_frame = frame;

		reverse_st_mrf_step(blocks, slit, frames, backgrounds, p.foreground_threshold);

//		auto edge = edge_image(frame);
		if (!plot_frame(frame, blocks, slit, 30 * 5))
			break;

//		std::cout << ".";
	}

	return 0;
}