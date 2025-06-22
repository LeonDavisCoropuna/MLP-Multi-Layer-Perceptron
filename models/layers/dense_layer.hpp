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

  Tensor forward(const Tensor &input) override
  {
    if (input.shape.size() != 1 || input.shape[0] != weights.shape[1])
    {
      throw std::runtime_error("Input shape mismatch in DenseLayer");
    }

    inputs = input;
    outputs = Tensor({weights.shape[0]}); // output_size

    // W·X + b
    for (int i = 0; i < weights.shape[0]; ++i)
    { // output_size
      float z = biases.data[i];
      for (int j = 0; j < weights.shape[1]; ++j)
      { // input_size
        z += weights.data[i * weights.shape[1] + j] * input.data[j];
      }
      outputs.data[i] = z;
    }

    // Aplicar activación
    if (dynamic_cast<Softmax *>(activation))
    {
      outputs = activation->activate_vector(outputs);
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
    if (targets != nullptr)
    {

      for (int i = 0; i < outputs.shape[0]; ++i)
      {
        float error = outputs.data[i] - targets->data[i];
        if (dynamic_cast<Softmax *>(activation))
        {
          deltas.data[i] = error; // Softmax + Cross-Entropy
        }
        else
        {
          deltas.data[i] = error * activation->derivative(outputs.data[i]);
        }
      }

      // Log: Deltas
      // Calcular gradiente respecto a la entrada (dL/dx = dL/dy * W^T)
      for (int i = 0; i < input_size(); ++i)
      {
        float sum = 0.0f;
        for (int j = 0; j < outputs.shape[0]; ++j)
        {
          sum += weights.data[j * input_size() + i] * deltas.data[j];
        }
        input_deltas.data[i] = sum;
      }

    }
    else if (next_layer != nullptr)
    {
      // Capa oculta
      const auto &next_deltas = next_layer->get_input_deltas(); // Usar input_deltas
      const auto &next_weights = next_layer->get_weights();


      for (int i = 0; i < outputs.shape[0]; ++i)
      {
        float sum = 0.0f;
        for (int j = 0; j < next_layer->output_size(); ++j)
        {
          if (next_layer->has_weights())
          {
            sum += next_weights.data[j * next_layer->input_size() + i] * next_deltas.data[j];
          }
          else
          {
            sum += next_deltas.data[i]; // Para capas sin pesos
          }
        }
        deltas.data[i] = sum * activation->derivative(outputs.data[i]);
      }

      // Log: Deltas
      // Calcular gradiente respecto a la entrada (dL/dx = dL/dy * W^T)
      for (int i = 0; i < input_size(); ++i)
      {
        float sum = 0.0f;
        for (int j = 0; j < outputs.shape[0]; ++j)
        {
          sum += weights.data[j * input_size() + i] * deltas.data[j];
        }
        input_deltas.data[i] = sum;
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
    for (int i = 0; i < weights.shape[0]; ++i)
    {
      for (int j = 0; j < weights.shape[1]; ++j)
      {
        accumulated_grad_weights.data[i * weights.shape[1] + j] += deltas.data[i] * inputs.data[j];
      }
      accumulated_grad_biases.data[i] += deltas.data[i];
    }
  }

  void apply_gradients(float batch_size) override
  {
    for (float &val : accumulated_grad_weights.data)
    {
      val /= batch_size;
    }
    for (float &val : accumulated_grad_biases.data)
    {
      val /= batch_size;
    }

    optimizer->update(weights, accumulated_grad_weights, biases, accumulated_grad_biases);
    zero_grad();
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