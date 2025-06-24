#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>

class Tensor
{
public:
  std::vector<int> shape;
  std::vector<float> data;

  Tensor() = default;
  Tensor(const std::vector<int> &shape) : shape(shape), data(calculate_size(shape), 0.0f) {}
  Tensor(const std::vector<int> &shape, const std::vector<float> &data) : shape(shape), data(data) {}

  size_t size() const
  {
    size_t total = 1;
    for (int dim : shape)
    {
      total *= dim;
    }
    return total;
  }

  Tensor reshape(const std::vector<int> &new_shape)
  {
    size_t new_size = 1;
    for (int dim : new_shape)
    {
      new_size *= dim;
    }
    if (new_size != size())
    {
      throw std::runtime_error("Reshape dimensions don't match tensor size");
    }
    shape = new_shape;
    return shape;
  }

  // New const version
  Tensor reshape(const std::vector<int> &new_shape) const
  {
    Tensor result(*this);      // Make a copy
    result.reshape(new_shape); // Call non-const version on the copy
    return result;
  }
  // Inicializa en ceros
  void init()
  {
    std::fill(data.begin(), data.end(), 0.0f);
  }

  // Inicialización aleatoria (uniforme)
  void rand_init(float lower = -0.1f, float upper = 0.1f)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower, upper);
    for (auto &val : data)
    {
      val = dist(gen);
    }
  }

  // Matmul (2D o batch de 3D)
  Tensor matmul(const Tensor &other) const
  {
    if (shape.size() == 2 && other.shape.size() == 2)
    {
      int M = shape[0], K = shape[1];
      int K2 = other.shape[0], N = other.shape[1];
      if (K != K2)
      {
        std::cerr << "K: " << K << " K2: " << K2 << std::endl;
        throw std::runtime_error("matmul: shape mismatch");
      }

      Tensor result({M, N});
      for (int i = 0; i < M; ++i)
      {
        for (int j = 0; j < N; ++j)
        {
          float sum = 0.0f;
          for (int k = 0; k < K; ++k)
          {
            sum += data[i * K + k] * other.data[k * N + j];
          }
          result.data[i * N + j] = sum;
        }
      }
      return result;
    }
    else if (shape.size() == 3 && other.shape.size() == 2)
    {
      int B = shape[0], L = shape[1], C = shape[2];
      int C2 = other.shape[0], C_out = other.shape[1];
      if (C != C2)
      {
        std::cerr << "C: " << C << " C2: " << C2 << std::endl;;
        throw std::runtime_error("matmul: shape mismatch");
      }

      Tensor result({B, L, C_out});
      for (int b = 0; b < B; ++b)
      {
        for (int i = 0; i < L; ++i)
        {
          for (int j = 0; j < C_out; ++j)
          {
            float sum = 0.0f;
            for (int k = 0; k < C; ++k)
            {
              sum += data[b * L * C + i * C + k] * other.data[k * C_out + j];
            }
            result.data[b * L * C_out + i * C_out + j] = sum;
          }
        }
      }
      return result;
    }
    else
    {
      throw std::runtime_error("matmul: unsupported dimensions");
    }
  }

  // Transpose solo para 3D: {B, L, C} → {B, C, L} o similar
  Tensor transpose(const std::vector<int> &axes) const
  {
    if (shape.size() != 3 || axes.size() != 3)
    {
      std::cout << "Shape: " << shape.size() << " Axes: " << axes.size() << std::endl;
      throw std::runtime_error("Only 3D transpose implemented");
    }

    std::vector<int> new_shape = {shape[axes[0]], shape[axes[1]], shape[axes[2]]};
    Tensor result(new_shape);

    int B = shape[0], L = shape[1], C = shape[2];

    for (int b = 0; b < B; ++b)
    {
      for (int l = 0; l < L; ++l)
      {
        for (int c = 0; c < C; ++c)
        {
          int src_idx = b * L * C + l * C + c;

          int i[] = {b, l, c};
          int new_b = i[axes[0]];
          int new_l = i[axes[1]];
          int new_c = i[axes[2]];

          int dst_idx = new_b * new_shape[1] * new_shape[2] + new_l * new_shape[2] + new_c;
          result.data[dst_idx] = data[src_idx];
        }
      }
    }

    return result;
  }
  Tensor transpose() const
  {
    if (shape.size() != 2)
    {
      throw std::runtime_error("Transpose solo soportado para matrices 2D");
    }

    Tensor result({shape[1], shape[0]});

    for (int i = 0; i < shape[0]; ++i)
    {
      for (int j = 0; j < shape[1]; ++j)
      {
        result.data[j * shape[0] + i] = data[i * shape[1] + j];
      }
    }

    return result;
  }

  Tensor permute(const std::vector<int> &dims) const
  {
    // Verifica que las dimensiones sean válidas
    if (dims.size() != shape.size())
    {
      throw std::runtime_error("Permute dimensions mismatch");
    }

    // Crea el nuevo tensor con las dimensiones permutadas
    std::vector<int> new_shape;
    for (int dim : dims)
    {
      new_shape.push_back(shape[dim]);
    }

    Tensor result(new_shape);

    // Calcula los strides originales
    std::vector<int> strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Copia los datos permutados
    std::vector<int> indices(shape.size(), 0);
    for (size_t i = 0; i < result.data.size(); ++i)
    {
      // Calcula la posición original
      int original_pos = 0;
      for (size_t j = 0; j < dims.size(); ++j)
      {
        original_pos += indices[j] * strides[dims[j]];
      }

      result.data[i] = data[original_pos];

      // Incrementa los índices
      int d = shape.size() - 1;
      while (d >= 0 && ++indices[d] == new_shape[d])
      {
        indices[d--] = 0;
      }
    }

    return result;
  }

  Tensor softmax(int axis) const
  {
    if (axis < 0 || axis >= shape.size())
      throw std::runtime_error("Softmax: axis out of range");

    Tensor result(shape);
    std::vector<int> strides(shape.size());
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    int outer = 1;
    for (int i = 0; i < axis; ++i)
      outer *= shape[i];

    int dim = shape[axis];

    int inner = 1;
    for (int i = axis + 1; i < shape.size(); ++i)
      inner *= shape[i];

    for (int o = 0; o < outer; ++o)
    {
      for (int i = 0; i < inner; ++i)
      {
        float max_val = -std::numeric_limits<float>::infinity();

        // Paso 1: encontrar el valor máximo para estabilidad numérica
        for (int d = 0; d < dim; ++d)
        {
          int idx = o * dim * inner + d * inner + i;
          max_val = std::max(max_val, data[idx]);
        }

        // Paso 2: calcular exponenciales
        float sum_exp = 0.0f;
        for (int d = 0; d < dim; ++d)
        {
          int idx = o * dim * inner + d * inner + i;
          result.data[idx] = std::exp(data[idx] - max_val);
          sum_exp += result.data[idx];
        }

        // Paso 3: normalizar
        for (int d = 0; d < dim; ++d)
        {
          int idx = o * dim * inner + d * inner + i;
          result.data[idx] /= sum_exp;
        }
      }
    }

    return result;
  }

  // Activaciones

  Tensor relu() const
  {
    Tensor result(*this); // Copia el tensor
    for (float &val : result.data)
    {
      val = std::max(0.0f, val);
    }
    return result;
  }

  // Versión 1: Devuelve el tamaño en una dimensión específica
  int size(int dim) const
  {
    if (dim < 0 || dim >= shape.size())
    {
      throw std::out_of_range("Invalid dimension");
    }
    return shape[dim];
  }

  Tensor slice(int dim, int start, int end) const
  {
    if (dim < 0 || dim >= shape.size())
    {
      throw std::out_of_range("Invalid dimension");
    }
    if (start < 0 || end > shape[dim] || start >= end)
    {
      throw std::out_of_range("Invalid slice range");
    }

    // Crear nuevo shape
    std::vector<int> new_shape = shape;
    new_shape[dim] = end - start;

    // Calcular strides y offsets
    int stride = std::accumulate(shape.begin() + dim + 1, shape.end(), 1, std::multiplies<int>());
    int offset = start * stride;

    // Crear nuevo tensor
    Tensor result(new_shape);
    int total_elements = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());

    // Copiar datos
    for (int i = 0; i < total_elements; ++i)
    {
      // Mapear índice nuevo a índice original
      // (esto requiere una función de indexación más compleja en casos generales)
      result.data[i] = data[offset + i];
    }

    return result;
  }

  // === Operadores entre tensores ===

  Tensor operator+(const Tensor &other) const
  {
    if (shape != other.shape)
      throw std::runtime_error("operator+: shape mismatch");
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i)
      result.data[i] = data[i] + other.data[i];
    return result;
  }

  Tensor operator-(const Tensor &other) const
  {
    if (shape != other.shape)
      throw std::runtime_error("operator-: shape mismatch");
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i)
      result.data[i] = data[i] - other.data[i];
    return result;
  }

  Tensor operator*(const Tensor &other) const
  {
    if (shape != other.shape)
      throw std::runtime_error("operator*: shape mismatch");
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i)
      result.data[i] = data[i] * other.data[i];
    return result;
  }

  Tensor operator/(const Tensor &other) const
  {
    if (shape != other.shape)
      throw std::runtime_error("operator/: shape mismatch");
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
      if (other.data[i] == 0.0f)
        throw std::runtime_error("Division by zero");
      result.data[i] = data[i] / other.data[i];
    }
    return result;
  }
  Tensor operator/(float scalar) const
  {
    Tensor result(shape);
    if (scalar == 0.0f)
      throw std::runtime_error("Division by zero");
    for (size_t i = 0; i < data.size(); ++i)
    {
      result.data[i] = data[i] / scalar;
    }
    return result;
  }
  void print_shape()
  {
    for (int i = 0; i < shape.size(); i++)
    {

      std::cout << shape[i] << ", ";
    }
  }

private:
  static size_t calculate_size(const std::vector<int> &shape)
  {
    size_t size = 1;
    for (int dim : shape)
    {
      size *= dim;
    }
    return size;
  }
};
