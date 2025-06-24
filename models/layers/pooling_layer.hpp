#pragma once
#include "layer.hpp"
#include <vector>
#include <stdexcept>
#include "../../utils/tensor.hpp"

enum class PoolingType
{
  MAX,
  AVG,
  MIN
};

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
  PoolingType pooling_type; // Tipo de pooling (MAX, AVG o MIN)

  void max_pooling_forward(int batch_size)
  {
    int in_spatial = in_height * in_width;
    int out_spatial = out_height * out_width;

    for (int b = 0; b < batch_size; ++b)
    {
      for (int c = 0; c < channels; ++c)
      {
        for (int oh = 0; oh < out_height; ++oh)
        {
          for (int ow = 0; ow < out_width; ++ow)
          {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_index = -1;

            for (int kh = 0; kh < kernel_size; ++kh)
            {
              for (int kw = 0; kw < kernel_size; ++kw)
              {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                if (ih < in_height && iw < in_width)
                {
                  int input_index = b * channels * in_spatial + c * in_spatial + ih * in_width + iw;
                  float val = inputs.data[input_index];
                  if (val > max_val)
                  {
                    max_val = val;
                    max_index = ih * in_width + iw;
                  }
                }
              }
            }

            int out_index = b * channels * out_spatial + c * out_spatial + oh * out_width + ow;
            outputs.data[out_index] = max_val;
            max_indices.data[out_index] = max_index;
          }
        }
      }
    }
  }

  void avg_pooling_forward()
  {
    float kernel_area = kernel_size * kernel_size;
    for (int c = 0; c < channels; ++c)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          float sum = 0.0f;
          for (int kh = 0; kh < kernel_size; ++kh)
          {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
              int input_h = h * stride + kh;
              int input_w = w * stride + kw;
              sum += inputs.data[c * in_height * in_width + input_h * in_width + input_w];
            }
          }
          outputs.data[c * out_height * out_width + h * out_width + w] = sum / kernel_area;
        }
      }
    }
  }

  void min_pooling_forward()
  {
    for (int c = 0; c < channels; ++c)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          float min_val = std::numeric_limits<float>::infinity();
          int min_idx = 0;
          for (int kh = 0; kh < kernel_size; ++kh)
          {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
              int input_h = h * stride + kh;
              int input_w = w * stride + kw;
              float val = inputs.data[c * in_height * in_width + input_h * in_width + input_w];
              if (val < min_val)
              {
                min_val = val;
                min_idx = input_h * in_width + input_w;
              }
            }
          }
          outputs.data[c * out_height * out_width + h * out_width + w] = min_val;
          max_indices.data[c * out_height * out_width + h * out_width + w] = static_cast<float>(min_idx);
        }
      }
    }
  }

  void max_pooling_backward()
  {
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

  void avg_pooling_backward()
  {
    float kernel_area = kernel_size * kernel_size;
    for (int c = 0; c < channels; ++c)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          float delta = deltas.data[c * out_height * out_width + h * out_width + w] / kernel_area;
          for (int kh = 0; kh < kernel_size; ++kh)
          {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
              int input_h = h * stride + kh;
              int input_w = w * stride + kw;
              input_deltas.data[c * in_height * in_width + input_h * in_width + input_w] += delta;
            }
          }
        }
      }
    }
  }

  void min_pooling_backward()
  {
    // Similar a max_pooling_backward pero usando los índices de mínimos
    for (int c = 0; c < channels; ++c)
    {
      for (int h = 0; h < out_height; ++h)
      {
        for (int w = 0; w < out_width; ++w)
        {
          int min_idx = static_cast<int>(max_indices.data[c * out_height * out_width + h * out_width + w]);
          input_deltas.data[c * in_height * in_width + min_idx] =
              deltas.data[c * out_height * out_width + h * out_width + w];
        }
      }
    }
  }

public:
  PoolingLayer(int channels, int in_height, int in_width, int kernel_size = 2, int stride = 1, PoolingType type = PoolingType::MAX)
      : channels(channels), in_height(in_height), in_width(in_width), kernel_size(kernel_size),
        stride(stride), pooling_type(type), training(true)
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

    if (input.shape.size() != 4 || input.shape[1] != channels || input.shape[2] != in_height || input.shape[3] != in_width)
    {
      std::cerr << "[PoolingLayer Error] Input shape mismatch in forward\n";
      throw std::runtime_error("Expected input shape [B, C, H, W]");
    }

    int batch_size = input.shape[0];

    // Redimensionar tensores para batches
    outputs = Tensor({batch_size, channels, out_height, out_width});
    deltas = Tensor({batch_size, channels, out_height, out_width});
    input_deltas = Tensor({batch_size, channels, in_height, in_width});
    max_indices = Tensor({batch_size, channels, out_height, out_width}); // Solo usado en max pooling

    // Realizar pooling para cada muestra en el batch
    switch (pooling_type)
    {
    case PoolingType::MAX:
      max_pooling_forward(batch_size);
      break;
    case PoolingType::AVG:
      // avg_pooling_forward(batch_size);
      break;
    case PoolingType::MIN:
      // min_pooling_forward(batch_size);
      break;
    default:
      throw std::runtime_error("Unknown pooling type");
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
    switch (pooling_type)
    {
    case PoolingType::MAX:
      max_pooling_backward();
      break;
    case PoolingType::AVG:
      avg_pooling_backward();
      break;
    case PoolingType::MIN:
      min_pooling_backward();
      break;
    default:
      throw std::runtime_error("Unknown pooling type in backward pass");
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