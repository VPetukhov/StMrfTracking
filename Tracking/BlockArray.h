#pragma once

#include <cstdlib>
#include <vector>

#include "opencv2/opencv.hpp"

namespace Tracking
{
	class BlockArray
	{
	public:
		using id_t = int;

		class Block
		{
		public:
			const size_t start_y;
			const size_t start_x;
			const size_t end_y;
			const size_t end_x;
			id_t object_id;

			Block(size_t start_y, size_t start_x, size_t height, size_t width);
			const cv::Range x_coords() const;
			const cv::Range y_coords() const;
		};

		class Slit
		{
		public:
			enum Direction
			{
				UP,
				DOWN
			};

		private:
			std::vector<size_t> _block_xs;
			size_t _block_y;
			Direction _direction;

		public:
			std::vector<size_t> block_xs() const;
			size_t block_y() const;
			Direction direction() const;
			Slit(size_t slit_y, size_t slit_x_start, size_t slit_x_end, size_t block_width, size_t block_height, Direction direction);
		};

	private:
		std::vector<Block> _blocks;

	public:
		static const int cv_id_t = cv::DataType<id_t>::type;

		const size_t block_height;
		const size_t block_width;

		const size_t height;
		const size_t width;

	public:
		Block& at(size_t index);
		Block& at(size_t row, size_t col);
		void set_object_ids(cv::Mat object_ids);

		const Block& at(size_t index) const;
		const Block& at(size_t row, size_t col) const;
		const Block& at(cv::Point coords) const;
		size_t index(size_t row, size_t col) const;
		bool valid_coords(const cv::Point& coords) const;
		bool valid_coords(long row, long col) const;

		cv::Mat pixel_object_map() const;
		cv::Mat object_map() const;

		BlockArray(size_t height, size_t width, size_t block_height, size_t block_width);
	};
}

