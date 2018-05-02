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

	double average(const cv::Scalar& scalar)
	{
		double res = 0;
		int n_vals = scalar.rows * scalar.cols;
		for (size_t i = 0; i < n_vals; ++i)
		{
			res += scalar.val[i];
		}

		return res / n_vals;
	}

	void show_image(const cv::Mat &img, int time)
	{
		namedWindow("edges", 1);
		imshow("edges", img);
		if(waitKey(time) >= 0)
			return;
	}
}
