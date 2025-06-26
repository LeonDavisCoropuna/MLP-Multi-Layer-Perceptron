#include "layer.hpp"
#include "../../utils/activations.hpp"
#include "../../utils/optimizer.hpp"
#include "../../utils/tensor.hpp"
#include <stdexcept>
#include <iomanip>

class DenseLayer : public Layer
{
private:
  Tensor weights;      // Forma: [output_size, input_size]
  Tensor biases;       // Forma: [output_size]
  Tensor deltas;       // Forma: [output_size]
  Tensor input_deltas; // Gradiente respecto a la entrada (dL/dx)
  Tensor outputs;      // Forma: [output_size]
  Tensor inputs;       // Forma: [input_size]

  ActivationFunction *activation;
  Optimizer *optimizer;
  bool training;

  Tensor accumulated_grad_weights; // Forma: [output_size, input_size]
  Tensor accumulated_grad_biases;  // Forma: [output_size]

public:
  DenseLayer(int input_size, int output_size,
             ActivationFunction *_activation, Optimizer *_optimizer)
      : activation(_activation), optimizer(_optimizer), training(true)
  {

    // Inicialización de pesos y biases
    weights = Tensor({output_size, input_size});
    biases = Tensor({output_size});

    // Inicialización He para los pesos
    float stddev = sqrt(2.0f / input_size);
    for (int i = 0; i < output_size; ++i)
    {
      for (int j = 0; j < input_size; ++j)
      {
        weights.data[i * input_size + j] = std::normal_distribution<float>(0.0f, stddev)(Layer::gen);
      }
      biases.data[i] = 0.01f; // Pequeño bias inicial
    }

    // Inicializar tensores auxiliares
    deltas = Tensor({output_size});
    outputs = Tensor({output_size});
    inputs = Tensor({input_size});
    input_deltas = Tensor({input_size});
    accumulated_grad_weights = Tensor({output_size, input_size});
    accumulated_grad_biases = Tensor({output_size});
  }

  Tensor forward(const std::vector<Tensor> &input_) override
  {
    Tensor input = input_[0];
    if (input.shape.size() != 2 || input.shape[1] != weights.shape[1])
      throw std::runtime_error("Input shape mismatch in DenseLayer forward");

    inputs = input;
    int batch_size = input.shape[0];
    int input_size = input.shape[1];
    int output_size = weights.shape[0];

    outputs = Tensor({batch_size, output_size});

    for (int b = 0; b < batch_size; ++b)
    {
      for (int i = 0; i < output_size; ++i)
      {
        float sum = biases.data[i];
        for (int j = 0; j < input_size; ++j)
        {
          sum += weights.data[i * input_size + j] * input.data[b * input_size + j];
        }
        outputs.data[b * output_size + i] = sum;
      }
    }

    // Activación
    if (dynamic_cast<Softmax *>(activation))
    {
      outputs = activation->activate(outputs);
    }
    else
    {
      for (float &val : outputs.data)
      {
        val = activation->activate(val);
      }
    }

    return outputs;
  }

  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    int batch_size = outputs.shape[0];
    int output_size = outputs.shape[1];
    int input_size_ = inputs.shape[1];

    deltas = Tensor({batch_size, output_size});
    input_deltas = Tensor({batch_size, input_size_});

    if (targets)
    {
      for (int b = 0; b < batch_size; ++b)
      {
        for (int i = 0; i < output_size; ++i)
        {
          int idx = b * output_size + i;
          float error = outputs.data[idx] - targets->data[idx];

          if (dynamic_cast<Softmax *>(activation))
          {
            deltas.data[idx] = error;
          }
          else
          {
            deltas.data[idx] = error * activation->derivative(outputs.data[idx]);
          }
          accumulated_grad_biases.data[i] += deltas.data[idx];
        }
      }
    }
    else if (next_layer)
    {
      const auto &next_deltas = next_layer->get_input_deltas(); // [B, next_out]
      const auto &next_weights = next_layer->get_weights();     // [next_out, curr_out]

      for (int b = 0; b < batch_size; ++b)
      {
        for (int i = 0; i < output_size; ++i)
        {
          float sum = 0.0f;
          for (int j = 0; j < next_layer->output_size(); ++j)
          {
            if (next_layer->has_weights())
            {
              sum += next_weights.data[j * output_size + i] * next_deltas.data[b * next_layer->output_size() + j];
            }
            else
            {
              sum += next_deltas.data[b * output_size + i];
            }
          }

          int idx = b * output_size + i;
          deltas.data[idx] = sum * activation->derivative(outputs.data[idx]);
          accumulated_grad_biases.data[i] += deltas.data[idx];
        }
      }
    }

    // Gradientes de pesos: W += deltasᵗ · inputs
    for (int b = 0; b < batch_size; ++b)
    {
      for (int i = 0; i < output_size; ++i)
      {
        for (int j = 0; j < input_size_; ++j)
        {
          accumulated_grad_weights.data[i * input_size_ + j] +=
              deltas.data[b * output_size + i] * inputs.data[b * input_size_ + j];
        }
      }
    }

    // Gradientes respecto a la entrada (para capa anterior)
    for (int b = 0; b < batch_size; ++b)
    {
      for (int i = 0; i < input_size_; ++i)
      {
        float sum = 0.0f;
        for (int j = 0; j < output_size; ++j)
        {
          sum += weights.data[j * input_size_ + i] * deltas.data[b * output_size + j];
        }
        input_deltas.data[b * input_size_ + i] = sum;
      }
    }
  }

  void update_weights() override
  {
    Tensor gradients_weights({weights.shape[0], weights.shape[1]});
    Tensor gradients_biases({biases.shape[0]});

    for (int i = 0; i < weights.shape[0]; ++i)
    {
      for (int j = 0; j < weights.shape[1]; ++j)
      {
        gradients_weights.data[i * weights.shape[1] + j] = deltas.data[i] * inputs.data[j];
      }
      gradients_biases.data[i] = deltas.data[i];
    }

    optimizer->update(weights, gradients_weights, biases, gradients_biases);
  }

  void accumulate_gradients() override
  {
  }

  void apply_gradients(float batch_size) override
  {
    accumulated_grad_weights = accumulated_grad_weights / batch_size;
    accumulated_grad_biases = accumulated_grad_biases / batch_size;
    Tensor ww = accumulated_grad_weights;
    optimizer->update(weights, accumulated_grad_weights, biases, accumulated_grad_biases);
  }

  void zero_grad() override
  {
    std::fill(deltas.data.begin(), deltas.data.end(), 0.0f);
    std::fill(accumulated_grad_weights.data.begin(), accumulated_grad_weights.data.end(), 0.0f);
    std::fill(accumulated_grad_biases.data.begin(), accumulated_grad_biases.data.end(), 0.0f);
  }

  // Métodos de acceso
  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_deltas() const override { return deltas; }
  const Tensor &get_weights() const override { return weights; }

  void set_weights(const Tensor &new_weights) override
  {
    if (new_weights.shape != weights.shape)
    {
      throw std::runtime_error("Weight shape mismatch in set_weights");
    }
    weights = new_weights;
  }

  int input_size() const override { return weights.shape[1]; }
  int output_size() const override { return weights.shape[0]; }
  bool has_weights() const override { return true; }
  const Tensor &get_input_deltas() const { return input_deltas; }
  void set_training(bool is_training) override
  {
    training = is_training;
  }
};