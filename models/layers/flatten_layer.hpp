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
  Tensor forward(const Tensor &input) override
  {
    input_shape = input.shape;
    outputs = input;

    outputs.reshape({static_cast<int>(input.size())});
    return outputs;
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    if (!next_layer)
      return;

    // Obtenemos los deltas de la capa siguiente
    const Tensor &next_deltas = next_layer->get_input_deltas();

    // Copiamos los deltas
    deltas = next_deltas;

    // Verificación de dimensiones
    if (deltas.size() != outputs.size())
    {
      throw std::runtime_error(
          "Deltas size mismatch in FlattenLayer backward. Expected: " +
          std::to_string(outputs.size()) + ", got: " +
          std::to_string(deltas.size()));
    }

    input_deltas = deltas;
    input_deltas.reshape(input_shape); // Recuperamos la forma original (ej: [32, 5, 5])
  }

  void update_weights() override {}
  void accumulate_gradients() override {}
  void apply_gradients(float) override {}
  void zero_grad() override {}

  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_weights() const override
  {
    throw std::runtime_error("FlattenLayer does not have weights");
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
