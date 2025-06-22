#pragma once
#include "layer.hpp"
#include <vector>
#include <stdexcept>
#include "../../utils/tensor.hpp"

class PoolingLayer : public Layer
{
private:
  int channels;        // Número de canales (C)
  int in_height;       // H_in: Alto de la entrada
  int in_width;        // W_in: Ancho de la entrada
  int out_height;      // H_out: Alto de la salida
  int out_width;       // W_out: Ancho de la salida
  int kernel_size;     // Tamaño del kernel (2x2)
  int stride;          // Stride (2)
  Tensor outputs;      // Salida: [C, H_out, W_out]
  Tensor inputs;       // Entrada: [C, H_in, W_in]
  Tensor deltas;       // Gradiente respecto a la salida (dL/dy)
  Tensor input_deltas; // Gradiente respecto a la entrada (dL/dx)
  Tensor max_indices;  // Índices de los valores máximos: [C, H_out, W_out]
  bool training;

public:
  PoolingLayer(int channels, int in_height, int in_width, int kernel_size = 2, int stride = 1)
      : channels(channels), in_height(in_height), in_width(in_width), kernel_size(kernel_size),
        stride(stride), training(true)
  {

    out_height = static_cast<int>(std::floor((in_height - kernel_size) / stride)) + 1;
    out_width = static_cast<int>(std::floor((in_width - kernel_size) / stride)) + 1;

    // Validar dimensiones
    if (out_height <= 0 || out_width <= 0)
    {
      std::cerr << "[PoolingLayer Error] Invalid output size:" << std::endl;
      std::cerr << "  in_height: " << in_height << ", in_width: " << in_width << std::endl;
      std::cerr << "  kernel_size: " << kernel_size << ", stride: " << stride << std::endl;
      std::cerr << "  out_height: " << out_height << ", out_width: " << out_width << std::endl;
      throw std::invalid_argument("Invalid input dimensions or kernel size for PoolingLayer");
    }

    if ((in_height - kernel_size) % stride != 0 || (in_width - kernel_size) % stride != 0)
    {
      std::cerr << "[PoolingLayer Error] Incompatible dimensions:" << std::endl;
      std::cerr << "  in_height: " << in_height << ", in_width: " << in_width << std::endl;
      std::cerr << "  kernel_size: " << kernel_size << ", stride: " << stride << std::endl;
      std::cerr << "  (in_height - kernel_size) % stride = " << (in_height - kernel_size) % stride << std::endl;
      std::cerr << "  (in_width - kernel_size) % stride = " << (in_width - kernel_size) % stride << std::endl;
      throw std::invalid_argument("Input dimensions not compatible with kernel size and stride");
    }

    // Inicializar tensores
    outputs = Tensor({channels, out_height, out_width});
    deltas = Tensor({channels, out_height, out_width});
    input_deltas = Tensor({channels, in_height, in_width});
    max_indices = Tensor({channels, out_height, out_width});
  }

  Tensor forward(const Tensor &input) override
  {
    inputs = input;
    outputs = Tensor({channels, out_height, out_width});
    max_indices = Tensor({channels, out_height, out_width});

    // Enhanced shape validation
    if (input.shape != std::vector<int>{channels, in_height, in_width})
    {
      std::cerr << "[PoolingLayer Error] Input shape mismatch:" << std::endl;
      std::cerr << "  Expected: [" << channels << ", " << in_height << ", " << in_width << "]" << std::endl;
      std::cerr << "  Received: [";
      for (size_t i = 0; i < input.shape.size(); ++i)
      {
        std::cerr << input.shape[i];
        if (i < input.shape.size() - 1)
          std::cerr << ", ";
      }
      std::cerr << "]" << std::endl;
      throw std::runtime_error("Input shape mismatch in PoolingLayer forward");
    }

    // Max-pooling
    for (int c = 0; c < channels; ++c)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          float max_val = -std::numeric_limits<float>::infinity();
          int max_idx = 0;
          for (int kh = 0; kh < kernel_size; ++kh)
          {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
              int input_h = h * stride + kh;
              int input_w = w * stride + kw;
              float val = inputs.data[c * in_height * in_width + input_h * in_width + input_w];
              if (val > max_val)
              {
                max_val = val;
                max_idx = input_h * in_width + input_w;
              }
            }
          }
          outputs.data[c * out_height * out_width + h * out_width + w] = max_val;
          max_indices.data[c * out_height * out_width + h * out_width + w] = static_cast<float>(max_idx);
        }
      }
    }
    return outputs;
  }

  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    if (targets != nullptr)
    {
      throw std::runtime_error("PoolingLayer cannot be the output layer");
    }

    if (!next_layer)
    {
      throw std::runtime_error("PoolingLayer requires next_layer for backward pass");
    }

    const auto &next_deltas = next_layer->get_input_deltas(); // Recibe dL/dy de la capa siguiente
    deltas = Tensor(outputs.shape);                           // dL/dy (mismo tamaño que outputs)
    input_deltas = Tensor(inputs.shape);                      // dL/dx (mismo tamaño que inputs)

    // Validar forma del gradiente recibido
    if (next_deltas.shape != outputs.shape)
    {
      throw std::runtime_error("Dimension mismatch in PoolingLayer backward");
    }

    // Copiar next_deltas a deltas
    for (size_t i = 0; i < deltas.data.size(); ++i)
    {
      deltas.data[i] = next_deltas.data[i];
    }

    // Calcular gradiente respecto a la entrada (dL/dx)
    for (int c = 0; c < channels; ++c)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          int max_idx = static_cast<int>(max_indices.data[c * out_height * out_width + h * out_width + w]);
          input_deltas.data[c * in_height * in_width + max_idx] =
              deltas.data[c * out_height * out_width + h * out_width + w];
        }
      }
    }
  }

  // Métodos de acceso
  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_deltas() const override { return deltas; }
  const Tensor &get_input_deltas() const override { return input_deltas; }
  const Tensor &get_weights() const override
  {
    static const Tensor empty;
    return empty; // Pooling no tiene pesos
  }

  void set_weights(const Tensor &) override {} // Pooling no tiene pesos

  int input_size() const override
  {
    return channels * in_height * in_width;
  }

  int output_size() const override
  {
    return channels * out_height * out_width;
  }

  bool has_weights() const override { return false; }

  void set_training(bool is_training) override
  {
    training = is_training;
  }

  // Métodos vacíos requeridos por la interfaz
  void update_weights() override {}
  void accumulate_gradients() override {}
  void apply_gradients(float) override {}
  void zero_grad() override {}
};