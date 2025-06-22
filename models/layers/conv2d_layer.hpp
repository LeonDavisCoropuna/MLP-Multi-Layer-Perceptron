#pragma once
#include "layer.hpp"
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
#include "../../utils/activations.hpp"
#include "../../utils/optimizer.hpp"

class Conv2DLayer : public Layer
{
private:
  int in_channels;                // C_in: Número de canales de entrada
  int out_channels;               // C_out: Número de filtros
  int kernel_size;                // Tamaño del kernel (k x k)
  int in_height;                  // H_in: Alto de la entrada
  int in_width;                   // W_in: Ancho de la entrada
  int out_height;                 // H_out: Alto de la salida
  int out_width;                  // W_out: Ancho de la salida
  Tensor weights;                 // Filtros: [C_out, C_in, kernel_size, kernel_size]
  Tensor biases;                  // Biases: [C_out]
  Tensor weight_grads;            // Gradientes de los pesos
  Tensor bias_grads;              // Gradientes de los biases
  Tensor outputs;                 // Salida: [C_out, H_out, W_out]
  Tensor inputs;                  // Entrada: [C_in, H_in, W_in]
  Tensor deltas;                  // Gradiente respecto a la salida (dL/dy)
  Tensor input_deltas;            // Gradiente respecto a la entrada (dL/dx)
  ActivationFunction *activation; // Función de activación (e.g., ReLU)
  Optimizer *optimizer;           // Optimizador (e.g., Adam)
  bool training;

public:
  Conv2DLayer(int in_channels, int out_channels, int kernel_size, int in_height, int in_width,
              ActivationFunction *act = nullptr, Optimizer *opt = nullptr)
      : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
        in_height(in_height), in_width(in_width), training(true), activation(act), optimizer(opt)
  {
    // Calcular dimensiones de salida
    out_height = in_height - kernel_size + 1;
    out_width = in_width - kernel_size + 1;

    // Validar dimensiones
    if (out_height <= 0 || out_width <= 0)
    {
      throw std::invalid_argument("Invalid input dimensions or kernel size");
    }

    // Inicializar tensores
    weights = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    biases = Tensor({out_channels});
    weight_grads = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    bias_grads = Tensor({out_channels});
    outputs = Tensor({out_channels, out_height, out_width});
    deltas = Tensor({out_channels, out_height, out_width});
    input_deltas = Tensor({in_channels, in_height, in_width});

    // Inicializar pesos con distribución uniforme
    float scale = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    for (size_t i = 0; i < weights.data.size(); ++i)
    {
      weights.data[i] = std::normal_distribution<float>(0.0f, scale)(Layer::gen);
    }
    for (size_t i = 0; i < biases.data.size(); ++i)
    {
      biases.data[i] = 0.0f;
    }
  }

  Tensor forward(const Tensor &input) override
  {
    inputs = input;
    outputs = Tensor({out_channels, out_height, out_width});

    // Validar forma de la entrada
    // Validar forma de la entrada
    if (input.shape != std::vector<int>{in_channels, in_height, in_width})
    {
      std::cerr << "[Conv2DLayer Error] Input shape mismatch in Conv2DLayer forward:" << std::endl;
      std::cerr << "  Esperado: [" << in_channels << ", " << in_height << ", " << in_width << "]" << std::endl;
      std::cerr << "  Recibido: [";
      for (size_t i = 0; i < input.shape.size(); ++i)
      {
        std::cerr << input.shape[i];
        if (i < input.shape.size() - 1)
          std::cerr << ", ";
      }
      std::cerr << "]" << std::endl;
      throw std::runtime_error("Input shape mismatch in Conv2DLayer forward");
    }

    // Convolución
    for (int co = 0; co < out_channels; ++co)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          float sum = biases.data[co];
          for (int ci = 0; ci < in_channels; ++ci)
          {
            for (int kh = 0; kh < kernel_size; ++kh)
            {
              for (int kw = 0; kw < kernel_size; ++kw)
              {
                int input_h = h + kh;
                int input_w = w + kw;
                sum += inputs.data[ci * in_height * in_width + input_h * in_width + input_w] *
                       weights.data[co * in_channels * kernel_size * kernel_size +
                                    ci * kernel_size * kernel_size + kh * kernel_size + kw];
              }
            }
          }
          outputs.data[co * out_height * out_width + h * out_width + w] =
              activation ? activation->activate(sum) : sum;
        }
      }
    }

    return outputs;
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    if (targets != nullptr)
    {
      // Capa de salida (poco común para Conv2D)
      if (targets->shape != outputs.shape)
      {
        throw std::runtime_error("Target shape mismatch in Conv2DLayer backward");
      }
      deltas = Tensor(outputs.shape);
      for (size_t i = 0; i < outputs.data.size(); ++i)
      {
        float error = outputs.data[i] - targets->data[i];
        deltas.data[i] = activation ? error * activation->derivative(outputs.data[i]) : error;
      }
    }
    else if (next_layer != nullptr)
    {
      // Capa oculta
      const auto &next_deltas = next_layer->get_input_deltas();
      deltas = Tensor(outputs.shape);

      // Calcular dL/dy (deltas) usando los input_deltas de la capa siguiente
      // Para capas como Pooling o Conv2D, esto requiere convolución completa
      if (next_layer->has_weights())
      {
        // Siguiente capa es densa
        const auto &next_weights = next_layer->get_weights();
        if (next_weights.shape.size() != 2)
        {
          throw std::runtime_error("Expected 2D weights in next layer");
        }
        for (int co = 0; co < out_channels; ++co)
        {
          for (int h = 0; h < out_height; ++h)
          {
            for (int w = 0; w < out_width; ++w)
            {
              float sum = 0.0f;
              for (size_t j = 0; j < next_deltas.data.size(); ++j)
              {
                sum += next_weights.data[j * next_layer->input_size() +
                                         (co * out_height * out_width + h * out_width + w)] *
                       next_deltas.data[j];
              }
              deltas.data[co * out_height * out_width + h * out_width + w] =
                  activation ? sum * activation->derivative(
                                         outputs.data[co * out_height * out_width + h * out_width + w])
                             : sum;
            }
          }
        }
      }
      else
      {
        // Siguiente capa sin pesos (e.g., Pooling)
        if (next_deltas.shape != outputs.shape)
        {
          throw std::runtime_error("Dimension mismatch in Conv2DLayer backward");
        }
        for (size_t i = 0; i < outputs.data.size(); ++i)
        {
          deltas.data[i] = activation ? next_deltas.data[i] * activation->derivative(outputs.data[i])
                                      : next_deltas.data[i];
        }
      }
    }
    else
    {
      throw std::runtime_error("Conv2DLayer requires next_layer or targets for backward");
    }

    // Calcular gradiente respecto a la entrada (dL/dx)
    input_deltas = Tensor({in_channels, in_height, in_width});
    for (int ci = 0; ci < in_channels; ++ci)
    {
      for (int h = 0; h < in_height; ++h)
      {
        for (int w = 0; w < in_width; ++w)
        {
          float sum = 0.0f;
          for (int co = 0; co < out_channels; ++co)
          {
            for (int kh = 0; kh < kernel_size; ++kh)
            {
              for (int kw = 0; kw < kernel_size; ++kw)
              {
                int out_h = h - kh;
                int out_w = w - kw;
                if (out_h >= 0 && out_h < out_height && out_w >= 0 && out_w < out_width)
                {
                  sum += deltas.data[co * out_height * out_width + out_h * out_width + out_w] *
                         weights.data[co * in_channels * kernel_size * kernel_size +
                                      ci * kernel_size * kernel_size + kh * kernel_size + kw];
                }
              }
            }
          }
          input_deltas.data[ci * in_height * in_width + h * in_width + w] = sum;
        }
      }
    }

    // Calcular gradientes de pesos y biases
    weight_grads = Tensor(weights.shape);
    bias_grads = Tensor(biases.shape);
    for (int co = 0; co < out_channels; ++co)
    {
      float bias_sum = 0.0f;
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          bias_sum += deltas.data[co * out_height * out_width + h * out_width + w];
        }
      }
      bias_grads.data[co] = bias_sum;

      for (int ci = 0; ci < in_channels; ++ci)
      {
        for (int kh = 0; kh < kernel_size; ++kh)
        {
          for (int kw = 0; kw < kernel_size; ++kw)
          {
            float weight_sum = 0.0f;
            for (int h = 0; h < out_height; ++h)
            {
              for (int w = 0; w < out_width; ++w)
              {
                weight_sum += deltas.data[co * out_height * out_width + h * out_width + w] *
                              inputs.data[ci * in_height * in_width + (h + kh) * in_width + (w + kw)];
              }
            }
            weight_grads.data[co * in_channels * kernel_size * kernel_size +
                              ci * kernel_size * kernel_size + kh * kernel_size + kw] = weight_sum;
          }
        }
      }
    }
  }

  void update_weights() override
  {
    if (optimizer)
    {
      optimizer->update(weights, weight_grads, biases, bias_grads);
    }
  }

  void zero_grad() override
  {
    weight_grads = Tensor(weight_grads.shape);
    bias_grads = Tensor(bias_grads.shape);
  }

  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_deltas() const override { return deltas; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
  const Tensor &get_weights() const override { return weights; }

  void set_weights(const Tensor &new_weights) override
  {
    if (new_weights.shape == weights.shape)
    {
      weights = new_weights;
    }
    else
    {
      throw std::runtime_error("Weight shape mismatch in Conv2DLayer");
    }
  }

  int input_size() const override
  {
    return in_channels * in_height * in_width;
  }

  int output_size() const override
  {
    return out_channels * out_height * out_width;
  }

  bool has_weights() const override { return true; }

  void set_training(bool is_training) override
  {
    training = is_training;
  }
  void accumulate_gradients() {}
  void apply_gradients(float batch_size) {}
};