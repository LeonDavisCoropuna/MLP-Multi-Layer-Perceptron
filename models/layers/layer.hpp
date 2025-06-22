#pragma once
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
class Layer
{
public:
  virtual ~Layer() = default;
  virtual Tensor forward(const Tensor &input) = 0;
  virtual void backward(const Tensor *targets = nullptr,
                        const Layer *next_layer = nullptr) = 0;
  virtual void update_weights() = 0;
  virtual void accumulate_gradients() = 0;
  virtual void apply_gradients(float batch_size) = 0;
  virtual void zero_grad() = 0;

  virtual const Tensor &get_outputs() const = 0;
  virtual const Tensor &get_weights() const = 0;
  virtual void set_weights(const Tensor &) = 0;

  virtual const Tensor &get_deltas() const = 0;
  virtual int input_size() const = 0;
  virtual int output_size() const = 0;
  virtual bool has_weights() const { return false; }
  virtual void set_training(bool is_trainig) = 0;
  virtual const Tensor &get_input_deltas() const = 0;
  static std::mt19937 gen;
};