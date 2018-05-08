#include <cmath>
#include "GcWrappers.h"
#include "BlockArray.h"

namespace Tracking
{
	int id_by_coords(int row, int col, int width)
	{
		return row * width + col;
	}

	int prob_to_score(double prob, double mult)
	{
		return -static_cast<int>(std::round(mult * std::log10(prob + 1e-20)));
	}

	gco_ptr_t gc_optimization_8_grid_graph(int width, int height, int n_labels)
	{
		auto gco = std::make_shared<GCoptimizationGeneralGraph>(height * width, n_labels);

		for (int row = 0; row < (height - 1); ++row)
		{
			for (int col = 0; col < (width - 1); ++col)
			{
				gco->setNeighbors(id_by_coords(row, col, width), id_by_coords(row + 1, col, width));
				gco->setNeighbors(id_by_coords(row, col, width), id_by_coords(row, col + 1, width));
				gco->setNeighbors(id_by_coords(row, col, width), id_by_coords(row + 1, col + 1, width));
				gco->setNeighbors(id_by_coords(row + 1, col, width), id_by_coords(row, col + 1, width));
			}
		}

		return gco;
	}

	void set_data_cost(const gco_ptr_t &gco, const cv::Mat &costs)
	{
		int max_v = 0, min_v = 50000000;
		for (int node_id = 0; node_id < costs.rows; ++node_id)
		{
			for (int label_id = 0; label_id < costs.cols; ++label_id)
			{
				auto cost = costs.at<int>(node_id, label_id);
				gco->setDataCost(node_id, label_id, cost);
				max_v = std::max(cost, max_v);
				min_v = std::min(cost, min_v);
			}
		}

//		std::cout << costs.cols << ": " << min_v << " " << max_v << std::endl;
	}

	void set_smooth_cost(const gco_ptr_t &gco, int n_labels, int penalty)
	{
		std::vector<int> smooth_cost(n_labels * n_labels, 0);
		for (int i = 0; i < n_labels; ++i)
		{
			for (int j = 0; j < n_labels; ++j)
			{
				if (i == j)
					continue;

				gco->setSmoothCost(i, j, penalty);
			}
		}
	}

	cv::Mat gco_to_label_map(const gco_ptr_t &gco, int height, int width)
	{
		if (width * height != gco->numSites())
			throw std::logic_error("Wrong size of label map: " + std::to_string(height) + "x" + std::to_string(width) +
					                       ". Number of sites: " + std::to_string(gco->numSites()));

		cv::Mat res = cv::Mat::zeros(height, width, BlockArray::cv_id_t);
		int max_label = 0;
		for (int row = 0; row < height; ++row)
		{
			for (int col = 0; col < width; ++col)
			{
				res.at<BlockArray::id_t>(row, col) = gco->whatLabel(id_by_coords(row, col, width));
				max_label = std::max(max_label, gco->whatLabel(id_by_coords(row, col, width)));
			}
		}

//		std::cout << "Max: " << max_label << std::endl;

		return res;
	}
}