#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>

class Tensor
{
public:
  std::vector<int> shape;
  std::vector<float> data;

  Tensor() = default;
  Tensor(const std::vector<int> &shape) : shape(shape), data(calculate_size(shape), 0.0f) {}
  Tensor(const std::vector<int> &shape, const std::vector<float> &data) : shape(shape), data(data) {}

  size_t size() const
  {
    size_t total = 1;
    for (int dim : shape)
    {
      total *= dim;
    }
    return total;
  }

  void reshape(const std::vector<int> &new_shape)
  {
    size_t new_size = 1;
    for (int dim : new_shape)
    {
      new_size *= dim;
    }
    if (new_size != size())
    {
      throw std::runtime_error("Reshape dimensions don't match tensor size");
    }
    shape = new_shape;
  }

private:
  static size_t calculate_size(const std::vector<int> &shape)
  {
    size_t size = 1;
    for (int dim : shape)
    {
      size *= dim;
    }
    return size;
  }
};
