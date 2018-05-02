#include <numeric>
#include "BlockArray.h"

namespace Tracking
{
	BlockArray::Block::Block(size_t start_y, size_t start_x, size_t height, size_t width)
		: start_y(start_y)
		, start_x(start_x)
		, end_y(start_y + height)
		, end_x(start_x + width)
		, object_id(0)
	{}

	const cv::Range BlockArray::Block::x_coords() const
	{
		return cv::Range(this->start_x, this->end_x);
	}

	const cv::Range BlockArray::Block::y_coords() const
	{
		return cv::Range(this->start_y, this->end_y);
	}

	BlockArray::BlockArray(size_t height, size_t width, size_t block_height, size_t block_width)
		: block_height(block_height)
		, block_width(block_width)
		, height(height)
		, width(width)
	{
		for (size_t row = 0; row < height; ++row)
		{
			for (size_t col = 0; col < width; ++col)
			{
				this->_blocks.emplace_back(row * block_height, col * block_width, block_height, block_width);
			}
		}
	}

	size_t BlockArray::index(size_t row, size_t col) const
	{
		return this->width * row + col;
	}

	const BlockArray::Block &BlockArray::at(size_t row, size_t col) const
	{
		return this->at(this->index(row, col));
	}

	const BlockArray::Block &BlockArray::at(size_t index) const
	{
		return this->_blocks.at(index);
	}

	BlockArray::Block &BlockArray::at(size_t row, size_t col)
	{
		return this->at(this->index(row, col));
	}

	BlockArray::Block &BlockArray::at(size_t index)
	{
		return this->_blocks.at(index);
	}

	void BlockArray::set_object_ids(cv::Mat object_ids)
	{
		if (object_ids.rows != this->height || object_ids.cols != this->width)
			throw std::logic_error("Wrong size of object ids: " + std::to_string(object_ids.rows) + "x" + std::to_string(object_ids.cols));

		if (object_ids.type() != cv::DataType<id_t>::type)
			throw std::logic_error("Wrong type of object ids: " + std::to_string(object_ids.type()));

		for (size_t row = 0; row < this->height; ++row)
		{
			for (size_t col = 0; col < this->width; ++col)
			{
				this->at(row, col).object_id = object_ids.at<id_t>(row, col);
			}
		}
	}

	cv::Mat BlockArray::pixel_object_map() const
	{
		cv::Mat res(this->height * this->block_height, this->width * this->block_width, cv::DataType<id_t>::type);
		for (auto const &block : this->_blocks)
		{
			for (int row = block.start_y; row < block.end_y; ++row)
			{
				for (int col = block.start_x; col < block.end_x; ++col)
				{
					res.at<id_t>(row, col) = block.object_id;
				}
			}
		}

		return res;
	}

	cv::Mat BlockArray::object_map() const
	{
		cv::Mat res(this->height, this->width, cv::DataType<id_t>::type);
		for (size_t row = 0; row < this->height; ++row)
		{
			for (size_t col = 0; col < this->width; ++col)
			{
				res.at<id_t>(row, col) = this->at(row, col).object_id;
			}
		}
		return res;
	}

	const BlockArray::Block &BlockArray::at(cv::Point coords) const
	{
		return this->at(coords.y, coords.x);
	}

	BlockArray::Slit::Slit(size_t slit_y, size_t slit_x_start, size_t slit_x_end, size_t block_width,
	                       size_t block_height, Direction direction)
		: _block_y(slit_y / block_height)
		, _block_xs(slit_x_end / block_width - slit_x_start / block_width)
		, _direction(direction)
	{
		std::iota(this->_block_xs.begin(), this->_block_xs.end(), slit_x_start / block_width);
	}

	std::vector<size_t> BlockArray::Slit::block_xs() const
	{
		return this->_block_xs;
	}

	size_t BlockArray::Slit::block_y() const
	{
		return this->_block_y;
	}

	BlockArray::Slit::Direction BlockArray::Slit::direction() const
	{
		return this->_direction;
	}
}