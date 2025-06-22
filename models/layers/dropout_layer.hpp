#include "layer.hpp"
#include <random>
#include <stdexcept>
#include "../../utils/tensor.hpp"

class DropoutLayer : public Layer
{
private:
  float dropout_rate;
  Tensor mask;
  Tensor outputs;
  Tensor inputs;
  Tensor deltas;       // Gradiente respecto a la salida (dL/dy)
  Tensor input_deltas; // Gradiente respecto a la entrada (dL/dx)
  bool training;

public:
  DropoutLayer(float rate) : dropout_rate(rate), training(true)
  {
    if (rate < 0.0f || rate >= 1.0f)
    {
      throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
  }

  Tensor forward(const Tensor &input) override
  {
    inputs = input;
    outputs = Tensor(input.shape);

    if (training)
    {
      mask = Tensor(input.shape);
      float keep_prob = 1.0f - dropout_rate;

      for (size_t i = 0; i < input.data.size(); ++i)
      {
        float rand_val = std::uniform_real_distribution<float>(0.0f, 1.0f)(Layer::gen);
        mask.data[i] = rand_val < keep_prob ? 1.0f / keep_prob : 0.0f;
        outputs.data[i] = input.data[i] * mask.data[i];
      }
    }
    else
    {
      outputs = input;
    }

    return outputs;
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    if (targets != nullptr)
    {
      throw std::runtime_error("DropoutLayer cannot be the output layer");
    }

    if (!next_layer)
    {
      throw std::runtime_error("DropoutLayer requires next_layer for backward pass");
    }

    const auto &next_deltas = next_layer->get_input_deltas(); // Recibe dL/dy de la capa siguiente
    deltas = Tensor(outputs.shape);                           // dL/dy (mismo tamaño que outputs)
    input_deltas = Tensor(inputs.shape);                      // dL/dx (mismo tamaño que inputs)

    if (next_deltas.shape != outputs.shape)
    {
      throw std::runtime_error("Dimension mismatch in DropoutLayer backward");
    }

    if (training)
    {
      // Calcular gradiente respecto a la salida (dL/dy) y entrada (dL/dx)
      for (size_t i = 0; i < deltas.data.size(); ++i)
      {
        deltas.data[i] = next_deltas.data[i];                      // dL/dy = gradiente recibido
        input_deltas.data[i] = next_deltas.data[i] * mask.data[i]; // dL/dx = dL/dy * mask
      }
    }
    else
    {
      // En modo evaluación, pasar el gradiente sin cambios
      deltas = next_deltas;
      input_deltas = next_deltas;
    }
  }

  // Métodos de acceso
  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_deltas() const override { return deltas; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
  const Tensor &get_weights() const override
  {
    static const Tensor empty;
    return empty; // Dropout no tiene pesos
  }

  void set_training(bool is_training) override
  {
    training = is_training;
  }

  void set_weights(const Tensor &) override {} // Dropout no tiene pesos

  int input_size() const override
  {
    return inputs.shape.empty() ? 0 : inputs.size();
  }

  int output_size() const override
  {
    return outputs.size();
  }

  // Métodos vacíos requeridos por la interfaz
  void update_weights() override {}
  void accumulate_gradients() override {}
  void apply_gradients(float) override {}
  void zero_grad() override {}

  bool has_weights() const override { return false; }
};