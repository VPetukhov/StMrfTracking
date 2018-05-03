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

	void show_image(const cv::Mat &img, int time, const std::string &wind_name)
	{
		namedWindow(wind_name, 1);
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
}
