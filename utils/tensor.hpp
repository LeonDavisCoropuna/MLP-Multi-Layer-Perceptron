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
    return *this;
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
  void init(float value)
  {
    std::fill(data.begin(), data.end(), value);
  }
  void init_xavier()
  {
    if (shape.size() < 2)
    {
      throw std::runtime_error("Xavier initialization requires at least 2D tensor");
    }

    // Consideramos solo la primera y segunda dimensión para fan_in y fan_out
    int fan_in = shape[shape.size() - 2];
    int fan_out = shape[shape.size() - 1];
    float limit = std::sqrt(6.0f / (fan_in + fan_out));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (auto &val : data)
    {
      val = dist(gen);
    }
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

  Tensor detach() const
  {
    return Tensor(this->shape, this->data); // copia sin vínculo a gradientes
  }
  void copy_(const Tensor &other)
  {
    if (shape != other.shape)
    {
      throw std::runtime_error("Tensor shapes must match for copy_");
    }
    data = other.data;
  }

  // Crea una matriz diagonal a partir de un vector 1D
  Tensor diag() const
  {
    if (shape.size() != 1)
    {
      throw std::runtime_error("diag() requires 1D tensor");
    }

    int n = shape[0];
    Tensor result({n, n});

    for (int i = 0; i < n; ++i)
    {
      result.at({i, i}) = data[i];
    }

    return result;
  }

  // Calcula la media a lo largo de una dimensión
  Tensor mean(int dim) const
  {
    if (dim < 0 || dim >= shape.size())
    {
      throw std::runtime_error("Invalid dimension for mean()");
    }

    std::vector<int> new_shape = shape;
    new_shape.erase(new_shape.begin() + dim);

    Tensor result(new_shape);
    int stride = 1;
    for (int i = dim + 1; i < shape.size(); ++i)
    {
      stride *= shape[i];
    }

    int repeat = shape[dim];
    int chunk_size = stride;
    int num_chunks = size() / (repeat * chunk_size);

    for (int i = 0; i < num_chunks; ++i)
    {
      for (int j = 0; j < chunk_size; ++j)
      {
        float sum = 0.0f;
        for (int k = 0; k < repeat; ++k)
        {
          sum += data[i * repeat * chunk_size + k * chunk_size + j];
        }
        result.data[i * chunk_size + j] = sum / repeat;
      }
    }

    return result;
  }

  // Producto externo entre dos tensores 1D
  Tensor outer(const Tensor &other) const
  {
    if (shape.size() != 1 || other.shape.size() != 1)
    {
      throw std::runtime_error("outer() requires 1D tensors");
    }

    Tensor result({shape[0], other.shape[0]});

    for (int i = 0; i < shape[0]; ++i)
    {
      for (int j = 0; j < other.shape[0]; ++j)
      {
        result.at({i, j}) = data[i] * other.data[j];
      }
    }

    return result;
  }

  // Acceso a elementos con índices
  float &at(const std::vector<int> &indices)
  {
    if (indices.size() != shape.size())
    {
      throw std::runtime_error("Invalid number of indices");
    }

    int pos = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
      if (indices[i] >= shape[i])
      {
        throw std::runtime_error("Index out of bounds");
      }
      pos += indices[i] * stride;
      stride *= shape[i];
    }

    return data[pos];
  }

  const float &at(const std::vector<int> &indices) const
  {
    // Versión const de at()
    if (indices.size() != shape.size())
    {
      throw std::runtime_error("Invalid number of indices");
    }

    int pos = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
      if (indices[i] >= shape[i])
      {
        throw std::runtime_error("Index out of bounds");
      }
      pos += indices[i] * stride;
      stride *= shape[i];
    }

    return data[pos];
  }

  Tensor greater_than(float threshold) const
  {
    Tensor result(this->shape);
    for (size_t i = 0; i < data.size(); ++i)
    {
      result.data[i] = (data[i] > threshold) ? 1.0f : 0.0f;
    }
    return result;
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
        std::cerr << "C: " << C << " C2: " << C2 << std::endl;
        ;
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
    // ✅ Caso NUEVO: 3D x 3D => {B, M, K} x {B, K, N} = {B, M, N}
    else if (shape.size() == 3 && other.shape.size() == 3)
    {
      int B1 = shape[0], M = shape[1], K = shape[2];
      int B2 = other.shape[0], K2 = other.shape[1], N = other.shape[2];
      if (B1 != B2 || K != K2)
      {
        std::cerr << "B1: " << B1 << " B2: " << B2 << " K: " << K << " K2: " << K2 << std::endl;
        throw std::runtime_error("matmul: shape mismatch in 3D x 3D");
      }

      Tensor result({B1, M, N});
      for (int b = 0; b < B1; ++b)
      {
        for (int i = 0; i < M; ++i)
        {
          for (int j = 0; j < N; ++j)
          {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
              float a = data[b * M * K + i * K + k];
              float b_val = other.data[b * K * N + k * N + j];
              sum += a * b_val;
            }
            result.data[b * M * N + i * N + j] = sum;
          }
        }
      }
      return result;
    }
    else if (shape.size() == 4 && other.shape.size() == 4)
    {
      int B = shape[0], H = shape[1], M = shape[2], K = shape[3];
      int B2 = other.shape[0], H2 = other.shape[1], K2 = other.shape[2], N = other.shape[3];

      if (B != B2 || H != H2 || K != K2)
        throw std::runtime_error("matmul: shape mismatch (4D x 4D)");

      Tensor result({B, H, M, N});
      for (int b = 0; b < B; ++b)
      {
        for (int h = 0; h < H; ++h)
        {
          for (int i = 0; i < M; ++i)
          {
            for (int j = 0; j < N; ++j)
            {
              float sum = 0.0f;
              for (int k = 0; k < K; ++k)
              {
                float a = data[b * H * M * K + h * M * K + i * K + k];
                float b_val = other.data[b * H * K * N + h * K * N + k * N + j];
                sum += a * b_val;
              }
              result.data[b * H * M * N + h * M * N + i * N + j] = sum;
            }
          }
        }
      }
      return result;
    }
    else if (shape.size() == 1 && other.shape.size() == 2)
    {
      int K = shape[0];
      int K2 = other.shape[0], N = other.shape[1];
      if (K != K2)
        throw std::runtime_error("matmul: shape mismatch for vector x matrix");

      Tensor result({N});
      for (int j = 0; j < N; ++j)
      {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
          sum += data[k] * other.data[k * N + j];
        }
        result.data[j] = sum;
      }
      return result;
    }
    else if (shape.size() == 2 && other.shape.size() == 1)
    {
      int M = shape[0], K = shape[1];
      int K2 = other.shape[0];
      if (K != K2)
        throw std::runtime_error("matmul: shape mismatch for matrix x vector");

      Tensor result({M});
      for (int i = 0; i < M; ++i)
      {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
          sum += data[i * K + k] * other.data[k];
        }
        result.data[i] = sum;
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
    if (axes.size() != shape.size())
    {
      throw std::runtime_error("transpose: axes size must match tensor dimension");
    }

    // Validar que sea una permutación válida
    std::vector<bool> seen(shape.size(), false);
    for (int axis : axes)
    {
      if (axis < 0 || axis >= (int)shape.size() || seen[axis])
      {
        throw std::runtime_error("transpose: invalid or duplicate axes");
      }
      seen[axis] = true;
    }

    // Nuevo shape
    std::vector<int> new_shape(shape.size());
    for (size_t i = 0; i < axes.size(); ++i)
    {
      new_shape[i] = shape[axes[i]];
    }

    // Calcular strides original y nuevo
    std::vector<int> old_strides(shape.size());
    std::vector<int> new_strides(shape.size());

    old_strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i)
    {
      old_strides[i] = old_strides[i + 1] * shape[i + 1];
    }

    new_strides[new_shape.size() - 1] = 1;
    for (int i = new_shape.size() - 2; i >= 0; --i)
    {
      new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    // Reordenar data
    Tensor result(new_shape);
    std::vector<int> old_indices(shape.size());
    for (size_t new_index = 0; new_index < result.data.size(); ++new_index)
    {
      int tmp = new_index;
      std::vector<int> new_indices(shape.size());
      for (size_t i = 0; i < shape.size(); ++i)
      {
        new_indices[i] = tmp / new_strides[i];
        tmp %= new_strides[i];
      }

      for (size_t i = 0; i < shape.size(); ++i)
      {
        old_indices[axes[i]] = new_indices[i];
      }

      int old_flat_index = 0;
      for (size_t i = 0; i < shape.size(); ++i)
      {
        old_flat_index += old_indices[i] * old_strides[i];
      }

      result.data[new_index] = data[old_flat_index];
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
  static Tensor concat(const Tensor &A, const Tensor &B, int axis)
  {
    if (A.shape.size() != 3 || B.shape.size() != 3)
      throw std::runtime_error("concat: Only 3D tensors are supported");

    int B1 = A.shape[0], D1 = A.shape[1], C1 = A.shape[2];
    int B2 = B.shape[0], D2 = B.shape[1], C2 = B.shape[2];

    if (B1 != B2)
      throw std::runtime_error("concat: batch size must match");

    if (axis == 1 && C1 != C2)
      throw std::runtime_error("concat: feature size (C) must match for axis=1");
    if (axis == 2 && D1 != D2)
      throw std::runtime_error("concat: sequence length (L) must match for axis=2");

    Tensor result;
    if (axis == 1)
    {
      int D_out = D1 + D2;
      result = Tensor({B1, D_out, C1});
      result.data.resize(B1 * D_out * C1);

      for (int b = 0; b < B1; ++b)
      {
        for (int d = 0; d < D1; ++d)
        {
          for (int c = 0; c < C1; ++c)
          {
            result.data[b * D_out * C1 + d * C1 + c] = A.data[b * D1 * C1 + d * C1 + c];
          }
        }
        for (int d = 0; d < D2; ++d)
        {
          for (int c = 0; c < C1; ++c)
          {
            result.data[b * D_out * C1 + (D1 + d) * C1 + c] = B.data[b * D2 * C1 + d * C1 + c];
          }
        }
      }
    }
    else if (axis == 2)
    {
      int C_out = C1 + C2;
      result = Tensor({B1, D1, C_out});
      result.data.resize(B1 * D1 * C_out);

      for (int b = 0; b < B1; ++b)
      {
        for (int d = 0; d < D1; ++d)
        {
          for (int c = 0; c < C1; ++c)
          {
            result.data[b * D1 * C_out + d * C_out + c] = A.data[b * D1 * C1 + d * C1 + c];
          }
          for (int c = 0; c < C2; ++c)
          {
            result.data[b * D1 * C_out + d * C_out + (C1 + c)] = B.data[b * D2 * C2 + d * C2 + c];
          }
        }
      }
    }
    else
    {
      throw std::runtime_error("concat: unsupported axis (only axis=1 or axis=2 supported)");
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
  Tensor squeeze() const
  {
    std::vector<int> new_shape;

    // Copiar solo las dimensiones que no son 1
    for (int dim : shape)
    {
      if (dim != 1)
      {
        new_shape.push_back(dim);
      }
    }

    // Caso especial: si todas las dimensiones eran 1, devolver un escalar (shape {1})
    if (new_shape.empty())
    {
      return Tensor({1}, data);
    }

    // No necesitamos modificar los datos, solo cambiar la forma
    return Tensor(new_shape, data);
  }

  Tensor squeeze(int dim) const
  {
    std::vector<int> new_shape;

    if (dim >= static_cast<int>(shape.size()))
    {
      throw std::out_of_range("Dimension out of range");
    }

    if (dim >= 0)
    {
      // Squeeze solo en la dimensión especificada
      if (shape[dim] != 1)
      {
        throw std::runtime_error("Can only squeeze dimension of size 1");
      }

      new_shape = shape;
      new_shape.erase(new_shape.begin() + dim);
    }
    else
    {
      // Squeeze todas las dimensiones de tamaño 1 (comportamiento por defecto)
      for (int d : shape)
      {
        if (d != 1)
          new_shape.push_back(d);
      }

      // Si todas eran 1, mantener al menos una dimensión
      if (new_shape.empty())
      {
        new_shape.push_back(1);
      }
    }

    return Tensor(new_shape, data);
  }
  Tensor unsqueeze(int dim) const
  {
    if (dim < 0 || dim > static_cast<int>(shape.size()))
    {
      throw std::out_of_range("Dimension out of range");
    }

    std::vector<int> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);

    return Tensor(new_shape, data);
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

  Tensor operator*(float scalar) const
  {
    Tensor result = *this;
    for (float &v : result.data)
      v *= scalar;
    return result;
  }
  // 🔹 Suma escalar
  Tensor operator+(float scalar) const
  {
    Tensor result = *this;
    for (float &v : result.data)
      v += scalar;
    return result;
  }

  // 🔹 Elevar al 0.5 (raíz cuadrada)
  Tensor sqrt() const
  {
    Tensor result = *this;
    for (float &v : result.data)
      v = std::sqrt(v);
    return result;
  }

  // 🔹 Media en ciertas dimensiones (simplificada para dim={0,2})
  Tensor mean(const std::vector<int> &dims) const
  {
    if (shape.size() != 3 || dims != std::vector<int>{0, 2})
      throw std::runtime_error("Only supports mean over dim {0,2} for shape [N,C,H*W]");

    int N = shape[0];
    int C = shape[1];
    int HW = shape[2];

    Tensor result({C});
    for (int c = 0; c < C; ++c)
    {
      float sum = 0.0f;
      for (int n = 0; n < N; ++n)
        for (int hw = 0; hw < HW; ++hw)
          sum += data[n * C * HW + c * HW + hw];

      result.data[c] = sum / (N * HW);
    }
    return result;
  }

  // 🔹 Varianza en ciertas dimensiones (dim={0,2})
  Tensor var(const std::vector<int> &dims, bool unbiased) const
  {
    if (shape.size() != 3 || dims != std::vector<int>{0, 2})
      throw std::runtime_error("Only supports var over dim {0,2} for shape [N,C,H*W]");

    Tensor mu = this->mean(dims);
    int N = shape[0];
    int C = shape[1];
    int HW = shape[2];
    int count = N * HW;
    Tensor result({C});

    for (int c = 0; c < C; ++c)
    {
      float sum_sq = 0.0f;
      for (int n = 0; n < N; ++n)
        for (int hw = 0; hw < HW; ++hw)
        {
          float val = data[n * C * HW + c * HW + hw];
          sum_sq += (val - mu.data[c]) * (val - mu.data[c]);
        }
      result.data[c] = sum_sq / (unbiased ? (count - 1) : count);
    }
    return result;
  }

  void print_shape()
  {
    for (int i = 0; i < shape.size(); i++)
    {

      std::cout << shape[i] << ", ";
    }
    std::cout<<std::endl;
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
