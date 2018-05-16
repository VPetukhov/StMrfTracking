#include "NightDetection.h"

using namespace cv;

namespace Tracking
{
	Mat log_filter(const Mat &img, double response_threshold)
	{
		double log_filter_arr[5][5] = {
				{-0.0239, -0.0460, -0.0499, -0.0460, -0.0239},
				{-0.0460, -0.0061,  0.0923, -0.0061, -0.0460},
				{-0.0499,  0.0923,  0.3182,  0.0923, -0.0499},
				{-0.0460, -0.0061,  0.0923, -0.0061, -0.0460},
				{-0.0239, -0.0460, -0.0499, -0.0460, -0.0239},
		};

		const cv::Mat log_filter = cv::Mat(5, 5, cv::DataType<double>::type, log_filter_arr);

		Mat res;
		filter2D(img, res, -1, log_filter);
		return res > response_threshold;
	}

	Mat detect_headlights(const Mat &img, double scale_factor, double response_threshold, double monochrome_threshold)
	{
		Mat input;
		cvtColor(img, input, CV_RGB2GRAY);

		if (input.type() != 5 && input.type() != 6)
		{
			input.convertTo(input, DataType<double>::type, 1. / 255.);
		}

		input = input > monochrome_threshold;

		Mat rescaled_img;
		Mat res = Mat::zeros(input.size(), CV_8UC1);
		double downscale_step = 1 / scale_factor;
		double downscale = 1;
		for (int scale_level = 1; scale_level < 5; ++scale_level)
		{
			downscale *= downscale_step;
			resize(input, rescaled_img, Size(), downscale, downscale);
			auto blobs = log_filter(rescaled_img, response_threshold);
			resize(blobs, blobs, input.size());
			res = max(res, blobs > 0);
		}

		res = min(res, input > 0);

		return res;
	}
}