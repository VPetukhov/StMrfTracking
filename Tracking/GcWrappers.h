#pragma once

#include <memory>
#include "GCoptimization.h"
#include "opencv2/opencv.hpp"

namespace Tracking
{
	using gco_ptr_t = std::shared_ptr<GCoptimizationGeneralGraph>;

	int id_by_coords(int row, int col, int width);
	int prob_to_score(double prob, double mult = 5);
	gco_ptr_t gc_optimization_8_grid_graph(int height, int width, int n_labels);
	void set_data_cost(const gco_ptr_t &gco, const cv::Mat &costs);
	void set_smooth_cost(const gco_ptr_t &gco, int n_labels, int penalty=1);
	cv::Mat gco_to_label_map(const gco_ptr_t &gco, int height, int width);
};

