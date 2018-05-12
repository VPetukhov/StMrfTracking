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
#include "Tracking/NightDetection.h"

using namespace cv;
using namespace Tracking;

static const std::string SCRIPT_NAME = "stmrf";
static const std::string WINDOW_NAME = "Detection";
static const size_t NA_VALUE = std::numeric_limits<size_t>::max();

struct Params
{
	double background_update_weight = 0.05;
	size_t block_width = 16;
	size_t block_height = 20;
	bool cant_parse = false;
	BlockArray::CaptureType capture_type = BlockArray::CaptureType::CROSS;
	double foreground_threshold = 0.05;
	int frame_freq=1;
	std::string out_dir = "";
	int reverse_history_size=5;
	std::string video_file = "";
	BlockArray::Line capture = BlockArray::Line(NA_VALUE, NA_VALUE, NA_VALUE, BlockArray::Line::UP);
	BlockArray::Line slit = BlockArray::Line(NA_VALUE, NA_VALUE, NA_VALUE, BlockArray::Line::UP);
};

void save_vehicle(const Mat &img, const Rect &b_box, const std::string &path, size_t img_id);

static void usage()
{
	std::cerr << SCRIPT_NAME <<":\n"
	          << "SYNOPSIS\n"
	          << "\t" << SCRIPT_NAME << " [options] -o out_dir slit_y slit_x_left slit_x_right capture_y capture_x_left capture_x_right video_file\n"
	          << "OPTIONS:\n"
	          << "\t-h height, --block-height: height of each block. Default: " << Params().block_height << "\n"
	          << "\t-w width, --block-width: width of each block. Default: " << Params().block_width << "\n"
	          << "\t-t threshold, --foreground-threshold: Threshold, used to distinguish background from foreground. Default: " << Params().foreground_threshold << "\n"
	          << "\t-o dir, --output-dir: Output directory. Default: " << Params().out_dir << "\n";
}

static Params parse_cmd_params(int argc, char **argv)
{
	Params params{};

	int option_index = 0;
	int c;
	static struct option long_options[] = {
			{"block-height", required_argument, nullptr, 'h'},
			{"block-width",  required_argument,	nullptr, 'w'},
			{"foreground-threshold", required_argument, nullptr, 't'},
			{nullptr, 0, nullptr, 0}
	};
	while ((c = getopt_long(argc, argv, "h:w:t:o:", long_options, &option_index)) != -1)
	{
		switch (c)
		{
			case 'h' :
				params.block_height = strtoul(optarg, nullptr, 10);
				break;
			case 'w' :
				params.block_width = strtoul(optarg, nullptr, 10);
				break;
			case 't' :
				params.foreground_threshold = strtod(optarg, nullptr);
				break;
			case 'o' :
				params.out_dir = std::string(optarg);
				break;
			default:
				std::cerr << SCRIPT_NAME << ": unknown arguments passed: '" << (char)c <<"'"  << std::endl;
				params.cant_parse = true;
				return params;
		}
	}

	if (params.out_dir.empty())
	{
		std::cerr << "Output directory must be supplied" << std::endl;
		params.cant_parse = true;
		return params;
	}

	if (optind > argc - 7)
	{
		params.cant_parse = true;
		return params;
	}

	params.slit.y = strtol(argv[optind++], nullptr, 10);
	params.slit.x_left = strtol(argv[optind++], nullptr, 10);
	params.slit.x_right = strtol(argv[optind++], nullptr, 10);
	params.capture.y = strtol(argv[optind++], nullptr, 10);
	params.capture.x_left = strtol(argv[optind++], nullptr, 10);
	params.capture.x_right = strtol(argv[optind++], nullptr, 10);
	params.video_file = argv[optind++];

	if (params.slit.y < params.capture.y)
	{
		params.slit.direction = BlockArray::Line::DOWN;
		params.capture.direction = BlockArray::Line::DOWN;
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

void draw_slit(Mat& img, const BlockArray &blocks, const BlockArray::Slit &slit, int thickness=1)
{
	auto block_start = blocks.at(slit.block_y(), slit.block_xs().front());
	auto block_end = blocks.at(slit.block_y(), slit.block_xs().back());

	line(img, Point(block_start.start_x, block_start.start_y), Point(block_end.end_x, block_start.start_y), CV_RGB(1, 0, 0), thickness);
	line(img, Point(block_start.start_x, block_start.end_y), Point(block_end.end_x, block_start.end_y), CV_RGB(1, 0, 0), thickness);
}

bool plot_frame(const Mat &frame, const BlockArray &blocks, const BlockArray::Slit &slit, const BlockArray::Line &capture, int delay=30)
{
//	auto plot_map = blocks.pixel_object_map();
	Mat plot_img = frame.clone();
//	Mat plot_img = heatmap(plot_map);

	for (auto const &bb : bounding_boxes(blocks))
	{
		rectangle(plot_img, bb.second, CV_RGB(0, 255, 0), 1);
	}
	draw_slit(plot_img, blocks, slit, 2);
	line(plot_img, Point(capture.x_left, capture.y), Point(capture.x_right, capture.y), CV_RGB(0, 0, 1), 2);

	imshow(WINDOW_NAME, plot_img);

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
	{
		usage();
		return 1;
	}

	VideoCapture cap(p.video_file);
	if(!cap.isOpened())  // check if we succeeded
	{
		std::cerr << "Can't open video file: " << p.video_file << std::endl;
		return 1;
	}

//	Mat background = estimate_background(p.video_file, 300, p.background_update_weight, 3);
//	Mat back_out;
//	background.convertTo(back_out, DataType<int>::type, 255);
//	imwrite("./bacgkround_d2.jpg", back_out);

	Mat background;
//	Mat back_in = imread("./bacgkround_night.jpg");
	Mat back_in = imread("./bacgkround_d2.jpg");
	back_in.convertTo(background, DataType<float>::type, 1 / 255.0);

	Mat frame;
	read_frame(cap, frame);

	BlockArray blocks(background.rows / p.block_height, background.cols / p.block_width, p.block_height, p.block_width);
//	draw_grid(frame, blocks);

	if (p.slit.y > frame.rows - frame.rows % p.block_height)
		throw std::logic_error("Slit y is too large: " + std::to_string(p.slit.y));

	BlockArray::Slit slit(p.slit, p.block_width, p.block_height);

//	draw_slit(frame, blocks, slit);
//	show_image(frame);

	namedWindow(WINDOW_NAME, 1);

	// Loop
	int i = 1;
	size_t out_id = 0;
	Mat old_frame;
	std::deque<Mat> frames, backgrounds;
	frames.push_back(frame.clone());
	backgrounds.push_back(background.clone());

	while (read_frame(cap, frame))
	{
		if (++i % p.frame_freq != 0)
			continue;

		update_background_weighted(background, frame, p.foreground_threshold, p.background_update_weight);

		frames.push_back(frame.clone());
		backgrounds.push_back(background.clone());

		if (frames.size() > p.reverse_history_size)
		{
			frames.pop_front();
			backgrounds.pop_front();
		}

		std::cout << "Step " << i << std::endl;

//		auto reg_vehicle_ids = register_vehicle_step(blocks, slit, frame, old_frame, background, p.foreground_threshold,
//		                                             p.capture, p.capture_type);

		auto reg_vehicle_ids = reverse_st_mrf_step(blocks, slit, frames, backgrounds, p.foreground_threshold,
		                                           p.capture, p.capture_type);
		auto b_boxes = bounding_boxes(blocks);
		for (auto id : reg_vehicle_ids)
		{
			save_vehicle(frame, b_boxes.at(id), p.out_dir,  out_id++);
		}

		old_frame = frame;

		if (!plot_frame(frame, blocks, slit, p.capture, 30 * p.frame_freq))
			break;
	}

	return 0;
}

void save_vehicle(const Mat &img, const Rect &b_box, const std::string &path, size_t img_id)
{
	Mat out_img = Mat(img, b_box) * 255;
	out_img.convertTo(out_img, CV_8UC3);
//	show_image(out_img);

	auto out_filename = path + "/v" + std::to_string(img_id) + ".png";
	if (!imwrite(out_filename, out_img))
		throw std::runtime_error("Can't write image: '" + out_filename + "'");
}
