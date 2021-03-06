#include <numeric>
#include "BlockArray.h"
#include "Utils.h"

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

		if (object_ids.type() != BlockArray::cv_id_t)
			throw std::logic_error("Wrong type of object ids: " + std::to_string(object_ids.type()));

		size_t block_ind = 0;
		for (size_t row = 0; row < this->height; ++row)
		{
			for (size_t col = 0; col < this->width; ++col)
			{
				this->_blocks[block_ind++].object_id = object_ids.at<id_t>(row, col);
			}
		}
	}

	cv::Mat BlockArray::pixel_object_map() const
	{
		cv::Mat res(this->height * this->block_height, this->width * this->block_width, BlockArray::cv_id_t);
		for (auto const &block : this->_blocks)
		{
			res(block.y_coords(), block.x_coords()) = block.object_id;
		}

		return res;
	}

	cv::Mat BlockArray::object_map() const
	{
		cv::Mat res(this->height, this->width, BlockArray::cv_id_t);
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

	bool BlockArray::valid_coords(const cv::Point& coords) const
	{
		return this->valid_coords(coords.y, coords.x);
	}

	bool BlockArray::valid_coords(long row, long col) const
	{
		return Tracking::valid_coords(row, col, this->height, this->width);
	}

	BlockArray::Slit::Slit(const Line &line, size_t block_width, size_t block_height)
		: _block_y(line.y / block_height)
		, _block_xs(line.x_right / block_width - line.x_left / block_width)
		, _direction(line.direction)
	{
		std::iota(this->_block_xs.begin(), this->_block_xs.end(), line.x_left / block_width);
	}

	std::vector<size_t> BlockArray::Slit::block_xs() const
	{
		return this->_block_xs;
	}

	size_t BlockArray::Slit::block_y() const
	{
		return this->_block_y;
	}

	BlockArray::Line::Direction BlockArray::Slit::direction() const
	{
		return this->_direction;
	}

	BlockArray::Line::Line(size_t y, size_t left_x, size_t right_x, Direction direction)
		: y(y)
		, x_left(left_x)
		, x_right(right_x)
		, direction(direction)
	{}

	BlockArray::Capture::Capture(size_t y, size_t left_x, size_t right_x, BlockArray::Line::Direction direction, CaptureType type)
		: Line(y, left_x, right_x, direction)
		, type(type)
	{}
}