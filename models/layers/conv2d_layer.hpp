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
  int in_channels;  // C_in: Número de canales de entrada
  int out_channels; // C_out: Número de filtros
  int kernel_size;  // Tamaño del kernel (k x k)
  int in_height;    // H_in: Alto de la entrada
  int in_width;     // W_in: Ancho de la entrada
  int out_height;   // H_out: Alto de la salida
  int out_width;    // W_out: Ancho de la salida
  int stride;
  int padding;
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
              int stride = 1, int padding = 0,
              ActivationFunction *act = nullptr, Optimizer *opt = nullptr)
      : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
        stride(stride), padding(padding),
        in_height(in_height), in_width(in_width), training(true), activation(act), optimizer(opt)
  {
    // Calcular dimensiones de salida

    out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

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
    // Guardar la entrada
    inputs = input;

    // Validar dimensiones
    if (input.shape.size() != 4 ||
        input.shape[1] != in_channels ||
        input.shape[2] != in_height ||
        input.shape[3] != in_width)
    {
      std::cerr << "[Conv2DLayer Error] Input shape mismatch\n";
      throw std::runtime_error("Input shape mismatch in Conv2DLayer forward");
    }

    int batch_size = input.shape[0];
    outputs = Tensor({batch_size, out_channels, out_height, out_width});

    // Atajo para offsets
    int inputHW = in_height * in_width;
    int kernelHW = kernel_size * kernel_size;
    int outputHW = out_height * out_width;

    for (int b = 0; b < batch_size; ++b)
    {
      for (int co = 0; co < out_channels; ++co)
      {
        for (int oh = 0; oh < out_height; ++oh)
        {
          for (int ow = 0; ow < out_width; ++ow)
          {
            float sum = biases.data[co];

            for (int ci = 0; ci < in_channels; ++ci)
            {
              for (int kh = 0; kh < kernel_size; ++kh)
              {
                for (int kw = 0; kw < kernel_size; ++kw)
                {
                  int ih = oh * stride + kh - padding;
                  int iw = ow * stride + kw - padding;

                  if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width)
                  {
                    // Índices planos para acceso
                    int input_index = b * in_channels * inputHW + ci * inputHW + ih * in_width + iw;
                    int weight_index = co * in_channels * kernelHW + ci * kernelHW + kh * kernel_size + kw;

                    sum += input.data[input_index] * weights.data[weight_index];
                  }
                }
              }
            }

            int output_index = b * out_channels * outputHW + co * outputHW + oh * out_width + ow;
            outputs.data[output_index] = activation ? activation->activate(sum) : sum;
          }
        }
      }
    }

    return outputs;
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    int batch_size = outputs.shape[0];
    int outputHW = out_height * out_width;
    int inputHW = in_height * in_width;
    int kernelHW = kernel_size * kernel_size;

    // 1. Calcular deltas (dL/dz)
    deltas = Tensor(outputs.shape); // [B, C_out, H_out, W_out]

    if (targets != nullptr)
    {
      if (targets->shape != outputs.shape)
        throw std::runtime_error("Target shape mismatch in Conv2DLayer backward");

      for (size_t i = 0; i < outputs.data.size(); ++i)
      {
        float error = outputs.data[i] - targets->data[i];
        deltas.data[i] = activation ? error * activation->derivative(outputs.data[i]) : error;
      }
    }
    else if (next_layer != nullptr)
    {
      const auto &next_deltas = next_layer->get_input_deltas();

      if (next_deltas.shape != outputs.shape)
        throw std::runtime_error("Dimension mismatch in Conv2DLayer backward");

      for (size_t i = 0; i < outputs.data.size(); ++i)
      {
        deltas.data[i] = activation ? next_deltas.data[i] * activation->derivative(outputs.data[i]) : next_deltas.data[i];
      }
    }
    else
    {
      throw std::runtime_error("Conv2DLayer requires next_layer or targets for backward");
    }

    // 2. Calcular input_deltas (dL/dx)
    input_deltas = Tensor(inputs.shape); // [B, C_in, H_in, W_in]

    for (int b = 0; b < batch_size; ++b)
    {
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
                  int oh = (h + padding - kh);
                  int ow = (w + padding - kw);

                  if (oh % stride == 0 && ow % stride == 0)
                  {
                    oh /= stride;
                    ow /= stride;

                    if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width)
                    {
                      int delta_idx = b * out_channels * outputHW + co * outputHW + oh * out_width + ow;
                      int weight_idx = co * in_channels * kernelHW + ci * kernelHW + kh * kernel_size + kw;
                      sum += deltas.data[delta_idx] * weights.data[weight_idx];
                    }
                  }
                }
              }
            }
            int input_idx = b * in_channels * inputHW + ci * inputHW + h * in_width + w;
            input_deltas.data[input_idx] = sum;
          }
        }
      }
    }

    for (int co = 0; co < out_channels; ++co)
    {
      float bias_sum = 0.0f;

      for (int b = 0; b < batch_size; ++b)
      {
        for (int oh = 0; oh < out_height; ++oh)
        {
          for (int ow = 0; ow < out_width; ++ow)
          {
            int delta_idx = b * out_channels * outputHW + co * outputHW + oh * out_width + ow;
            float delta = deltas.data[delta_idx];
            bias_sum += delta;

            for (int ci = 0; ci < in_channels; ++ci)
            {
              for (int kh = 0; kh < kernel_size; ++kh)
              {
                for (int kw = 0; kw < kernel_size; ++kw)
                {
                  int ih = oh * stride + kh - padding;
                  int iw = ow * stride + kw - padding;

                  if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width)
                  {
                    int input_idx = b * in_channels * inputHW + ci * inputHW + ih * in_width + iw;
                    int weight_idx = co * in_channels * kernelHW + ci * kernelHW + kh * kernel_size + kw;
                    weight_grads.data[weight_idx] += delta * inputs.data[input_idx];
                  }
                }
              }
            }
          }
        }
      }
      bias_grads.data[co] = bias_sum;
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
  void apply_gradients(float batch_size) override
  {
    // Escalar los gradientes por el tamaño del batch
    for (float &val : weight_grads.data)
      val /= batch_size;

    for (float &val : bias_grads.data)
      val /= batch_size;

    // Actualizar pesos y sesgos
    if (optimizer)
    {
      optimizer->update(weights, weight_grads, biases, bias_grads);
    }
  }
};