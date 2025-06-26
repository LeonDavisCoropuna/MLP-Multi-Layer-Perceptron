#pragma once
#include "../../utils/tensor.hpp"
#include "layer.hpp"

class FlattenLayer : public Layer
{
private:
  Tensor outputs;
  Tensor deltas;
  Tensor input_deltas;
  std::vector<int> input_shape; // Para restaurar en backward si es necesario
  bool training = false;

public:
  Tensor forward(const std::vector<Tensor> &input_) override
  {
    auto input = input_[0];
    input_shape = input.shape;
    if (input.shape.size() < 2)
    {
      throw std::runtime_error("FlattenLayer expects input with at least 2 dimensions (batch + features)");
    }

    int batch_size = input.shape[0];
    int features = 1;

    for (size_t i = 1; i < input.shape.size(); ++i)
    {
      features *= input.shape[i];
    }

    outputs = input;
    outputs.reshape({batch_size, features}); // [B, C*H*W]

    return outputs;
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    if (!next_layer)
      return;

    const Tensor &next_deltas = next_layer->get_input_deltas();

    // Validación
    if (next_deltas.shape.size() != 2 || next_deltas.size() != outputs.size())
    {
      throw std::runtime_error(
          "Deltas shape mismatch in FlattenLayer backward. Expected 2D flat tensor with same size as outputs.");
    }

    deltas = next_deltas;

    // Restauramos la forma original (ej: [B, C, H, W])
    input_deltas = deltas;
    input_deltas.reshape(input_shape);
  }

  void update_weights() override {}
  void accumulate_gradients() override {}
  void apply_gradients(float) override {}
  void zero_grad() override {}

  const Tensor &get_outputs() const override { return outputs; }

  const Tensor &get_weights() const override
  {
    static const Tensor empty;
    return empty; // Dropout no tiene pesos
  }

  void set_weights(const Tensor &) override
  {
    throw std::runtime_error("Cannot set weights on FlattenLayer");
  }

  const Tensor &get_deltas() const override { return deltas; }

  int input_size() const override { return input_shape.empty() ? 0 : Tensor(input_shape).size(); }
  int output_size() const override { return outputs.size(); }

  bool has_weights() const override { return false; }

  void set_training(bool is_training) override { training = is_training; }
  const Tensor &get_input_deltas() const { return input_deltas; }
};
