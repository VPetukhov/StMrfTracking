#include "NightDetection.h"

using namespace cv;

namespace Tracking
{
	Mat log_filter(const Mat &img, double response_threshold, int monochrome_threshold)
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
		Mat inp = img > monochrome_threshold;
		inp.convertTo(inp, DataType<double>::type, 1. / 255.);
		filter2D(inp, res, -1, log_filter);
		res.setTo(0, res < 0);

		return res > response_threshold;
	}

	Mat detect_headlights(const Mat &img, double scale_factor)
	{
		Mat rescaled_img;
		Mat res = Mat::zeros(img.size(), CV_8UC1);
		double downscale_step = 1 / scale_factor;
		double downscale = 1;
		for (int scale_level = 1; scale_level < 5; ++scale_level)
		{
			downscale *= downscale_step;
			resize(img, rescaled_img, Size(), downscale, downscale);
			auto blobs = log_filter(rescaled_img);
			resize(blobs, blobs, img.size());
			res = max(res, blobs > 0);
		}

		res.convertTo(res, CV_8U);

		connectedComponents(res, res);
		return res;
	}
}