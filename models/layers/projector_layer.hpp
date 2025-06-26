#pragma once
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
#include "layer.hpp"
class ProjectorLayer : public Layer
{
private:
  int in_channels;    // C_in
  int out_channels;   // C_out
  int token_channels; // D

  // Capas lineales
  Tensor linear1_weights; // (in_channels, token_channels) - query
  Tensor linear2_weights; // (token_channels, token_channels) - key
  Tensor linear3_weights; // (token_channels, out_channels) - value

  // Capa de downsample (opcional)
  Tensor downsample_weights;
  bool use_downsample;

  // BatchNorm
  Tensor bn_gamma;
  Tensor bn_beta;
  Tensor bn_running_mean;
  Tensor bn_running_var;

  // Caches
  Tensor cache;         // Almacena atención para debugging
  Tensor x_input_cache; // Almacena input original
  Tensor t_input_cache;
  Tensor outputs;
  // Gradientes
  Tensor gradients_linear1;
  Tensor gradients_linear2;
  Tensor gradients_linear3;
  Tensor gradients_downsample;
  Tensor gradients_bn_gamma;
  Tensor gradients_bn_beta;

  // Caches para backward
  Tensor x_norm; // Cache para BatchNorm
  Tensor t_q;    // Cache para queries de tokens
  Tensor x_q;    // Cache para queries de features

  // Deltas de salida
  Tensor input_deltas; // [N, HW, C_in]
  Tensor token_deltas; // [N, L, D]

public:
  ProjectorLayer(int in_channels, int out_channels, int token_channels)
      : in_channels(in_channels),
        out_channels(out_channels),
        token_channels(token_channels)
  {

    // Inicialización de pesos con Xavier normal
    linear1_weights = Tensor({in_channels, token_channels});
    linear2_weights = Tensor({token_channels, token_channels});
    linear3_weights = Tensor({token_channels, out_channels});

    linear1_weights.init_xavier();
    linear2_weights.init_xavier();
    linear3_weights.init_xavier();

    // Inicializar downsample si es necesario
    use_downsample = (in_channels != out_channels);
    if (use_downsample)
    {
      downsample_weights = Tensor({in_channels, out_channels});
      downsample_weights.init_xavier();
    }

    // Inicializar BatchNorm
    bn_gamma = Tensor({out_channels});
    bn_beta = Tensor({out_channels});
    bn_running_mean = Tensor({out_channels});
    bn_running_var = Tensor({out_channels});

    bn_gamma.init(1.0f); // Gamma inicializado a 1
    bn_beta.init(0.0f);  // Beta inicializado a 0
    bn_running_mean.init(0.0f);
    bn_running_var.init(1.0f);
  }

  Tensor forward(const std::vector<Tensor> &input_) override
  {
    // Verificar dimensiones
    // x: [N, HW, C_in]
    // t: [N, L, D]
    auto x = input_[0];
    auto t = input_[1];

    t_input_cache = t;
    x_input_cache = x;
    if (x.shape.size() != 3 || t.shape.size() != 3)
    {
      throw std::invalid_argument("Inputs must be 3D tensors");
    }

    int N = x.shape[0];
    int HW = x.shape[1];
    int C_in = x.shape[2];
    int L = t.shape[1];
    int D = t.shape[2];
    int tok_cha = token_channels;
    if (C_in != in_channels || D != token_channels)
    {
      throw std::invalid_argument("Channel dimensions mismatch");
    }

    x_input_cache = x; // Guardar para posible conexión residual

    // Paso 1: Proyectar feature map (query)
    Tensor x_q = x.matmul(linear1_weights); // [N, HW, D]

    // Paso 2: Proyectar tokens (key)
    Tensor t_q = t.matmul(linear2_weights); // [N, L, D]

    // Paso 3: Calcular atención
    Tensor t_q_transposed = t_q.transpose({0, 2, 1}); // [N, D, L]
    Tensor a = x_q.matmul(t_q_transposed);            // [N, HW, L]
    a = a.softmax(2);                                 // Softmax sobre dimensión L
    cache = a;                                        // Guardar atención para debugging

    // Paso 4: Proyectar tokens (value)
    Tensor t_v = t.matmul(linear3_weights); // [N, L, C_out]

    // Paso 5: Aplicar atención
    Tensor attended = a.matmul(t_v); // [N, HW, C_out]

    // Paso 6: Conexión residual (con downsample si es necesario)
    Tensor x_out;
    if (use_downsample)
    {
      Tensor x_down = x.matmul(downsample_weights); // [N, HW, C_out]
      x_out = x_down + attended;
    }
    else
    {
      x_out = x + attended;
    }

    // Paso 7: BatchNorm + ReLU
    // Transponer para BN: [N, HW, C_out] -> [N, C_out, HW]
    x_out = x_out.transpose({0, 2, 1});

    // Aplicar BatchNorm
    x_out = batch_norm_forward(x_out, bn_gamma, bn_beta,
                               bn_running_mean, bn_running_var);

    // Transponer de vuelta: [N, C_out, HW] -> [N, HW, C_out]
    x_out = x_out.transpose({0, 2, 1});

    // Aplicar ReLU
    x_out = x_out.relu();
    outputs = x_out;
    return x_out;
  }

public:
  Tensor batch_norm_forward(const Tensor &x, const Tensor &gamma, const Tensor &beta,
                            Tensor &running_mean, Tensor &running_var,
                            float eps = 1e-5, float momentum = 0.1)
  {
    int N = x.shape[0];
    int C = x.shape[1];
    int HW = x.shape[2];

    // Calcular media y varianza
    Tensor mean = x.mean({0, 2});     // [C]
    Tensor var = x.var({0, 2}, true); // [C]

    // Actualizar running stats
    running_mean = running_mean * (1 - momentum) + mean * momentum;
    running_var = running_var * (1 - momentum) + var * momentum;

    // Normalizar con broadcasting manual
    Tensor x_norm(x.shape);
    for (int n = 0; n < N; ++n)
    {
      for (int c = 0; c < C; ++c)
      {
        for (int hw = 0; hw < HW; ++hw)
        {
          int idx = n * C * HW + c * HW + hw;
          x_norm.data[idx] = (x.data[idx] - mean.data[c]) /
                             std::sqrt(var.data[c] + eps);
        }
      }
    }

    // Escalar y desplazar con broadcasting manual
    Tensor result(x.shape);
    for (int n = 0; n < N; ++n)
    {
      for (int c = 0; c < C; ++c)
      {
        for (int hw = 0; hw < HW; ++hw)
        {
          int idx = n * C * HW + c * HW + hw;
          result.data[idx] = gamma.data[c] * x_norm.data[idx] + beta.data[c];
        }
      }
    }

    return result;
  }

  // Placeholder for backward pass - would need to be implemented
  void backward(const Tensor *next_layer_deltas = nullptr,
                const Layer *next_layer = nullptr) override
  {
    // 1. Obtener los deltas de la siguiente capa
    Tensor delta_out;
    if (next_layer_deltas != nullptr)
    {
      delta_out = *next_layer_deltas;
    }
    else if (next_layer != nullptr)
    {
      delta_out = next_layer->get_input_deltas();
    }
    else
    {
      throw std::runtime_error("No gradient source provided for backward pass");
    }

    // 2. Backward a través de ReLU
    Tensor delta_relu = delta_out * (outputs.greater_than(0)); // Gradiente de ReLU

    // 3. Backward a través de BatchNorm
    // Transponer para BN backward: [N, HW, C_out] -> [N, C_out, HW]
    Tensor delta_bn = delta_relu.transpose({0, 2, 1});
    delta_bn = batch_norm_backward(delta_bn, bn_gamma, bn_beta,
                                   bn_running_mean, bn_running_var);
    // Transponer de vuelta: [N, C_out, HW] -> [N, HW, C_out]
    delta_bn = delta_bn.transpose({0, 2, 1});

    // 4. Separar gradientes de la conexión residual
    Tensor delta_attended = delta_bn;   // Gradiente de la parte de atención
    Tensor delta_x_residual = delta_bn; // Gradiente de la conexión residual

    // 5. Backward a través de la suma residual (downsample si existe)
    Tensor delta_x;
    if (use_downsample)
    {
      // Gradiente a través del downsample
      delta_x = delta_x_residual.matmul(downsample_weights.transpose({1, 0}));

      // Calcular gradiente de downsample_weights
      gradients_downsample = x_input_cache.transpose({0, 2, 1}).matmul(delta_x_residual).mean(0); // Promedio sobre el batch
    }
    else
    {
      delta_x = delta_x_residual;
    }

    // 6. Backward a través de la multiplicación atención-valor (a.matmul(t_v))
    // delta_a = delta_attended.matmul(t_v.transpose({0, 2, 1}))
    // delta_t_v = a.transpose({0, 2, 1}).matmul(delta_attended)

    // 7. Backward a través de linear3 (t.matmul(linear3_weights))
    Tensor t_v = x_input_cache.matmul(linear3_weights); // Necesitamos cachear esto en el forward
    Tensor delta_t = delta_attended.matmul(t_v.transpose({0, 2, 1}));
    gradients_linear3 = t_input_cache.transpose({0, 2, 1}).matmul(delta_attended).mean(0);

    // 8. Backward a través del softmax de atención
    Tensor delta_softmax = delta_t; // Esto necesita la derivada del softmax

    // 9. Backward a través de la multiplicación query-key (x_q.matmul(t_q_transposed))
    Tensor delta_x_q = delta_softmax.matmul(t_q.transpose({0, 2, 1}));
    Tensor delta_t_q_transposed = x_q.transpose({0, 2, 1}).matmul(delta_softmax);

    // 10. Backward a través de linear2 (t.matmul(linear2_weights))
    Tensor delta_t_q = delta_t_q_transposed.transpose({0, 2, 1});
    gradients_linear2 = t_input_cache.transpose({0, 2, 1}).matmul(delta_t_q).mean(0);
    Tensor delta_t_from_q = delta_t_q.matmul(linear2_weights.transpose({1, 0}));

    // 11. Backward a través de linear1 (x.matmul(linear1_weights))
    gradients_linear1 = x_input_cache.transpose({0, 2, 1}).matmul(delta_x_q).mean(0);
    Tensor delta_x_from_q = delta_x_q.matmul(linear1_weights.transpose({1, 0}));

    // 12. Combinar gradientes de ambas rutas
    input_deltas = delta_x + delta_x_from_q; // [N, HW, C_in]
    token_deltas = delta_t_from_q;           // [N, L, D]

    // Guardar gradientes para actualización
    gradients_linear1 = gradients_linear1;
    gradients_linear2 = gradients_linear2;
    gradients_linear3 = gradients_linear3;
    if (use_downsample)
    {
      gradients_downsample = gradients_downsample;
    }
    // gradients_bn_gamma = /* calcular gradiente de gamma */;
    // gradients_bn_beta = /* calcular gradiente de beta */;
  }

  // Placeholder for update_weights
  void update_weights() override
  {
    // if (optimizer)
    // {
    //   optimizer->update(weights_query, gradients_query, /*bias=*/weights_key, /*bias_grad=*/gradients_key);
    // }
  }

  // Placeholder for zero_grad
  void zero_grad() override
  {
    // gradients_query.init();
    // gradients_key.init();
  }

  Tensor batch_norm_backward(const Tensor &delta, const Tensor &gamma,
                             const Tensor &beta, Tensor &running_mean,
                             Tensor &running_var, float eps = 1e-5)
  {
    int N = delta.shape[0];
    int C = delta.shape[1];
    int HW = delta.shape[2];

    Tensor delta_gamma({C}); // 👈 correcto
    Tensor delta_beta({C});

    for (int c = 0; c < C; ++c)
    {
      float sum_gamma = 0.0f;
      float sum_beta = 0.0f;
      for (int n = 0; n < N; ++n)
      {
        for (int hw = 0; hw < HW; ++hw)
        {
          int idx = n * C * HW + c * HW + hw;
          sum_gamma += delta.data[idx] * x_norm.data[idx]; // x_norm debe estar cacheado
          sum_beta += delta.data[idx];
        }
      }
      delta_gamma.data[c] = sum_gamma;
      delta_beta.data[c] = sum_beta;
    }

    Tensor delta_out(delta.shape);
    float sqrt_var_eps = std::sqrt(running_var.data[0] + eps);
    for (int n = 0; n < N; ++n)
    {
      for (int c = 0; c < C; ++c)
      {
        for (int hw = 0; hw < HW; ++hw)
        {
          int idx = n * C * HW + c * HW + hw;
          delta_out.data[idx] = gamma.data[c] * delta.data[idx] / sqrt_var_eps;
        }
      }
    }

    gradients_bn_gamma = delta_gamma;
    gradients_bn_beta = delta_beta;

    return delta_out;
  }

  const Tensor &get_outputs() const override
  {
    static Tensor dummy_weights = Tensor({1});
    return dummy_weights;
  }
  const Tensor &get_weights() const override
  {
    static Tensor dummy_weights = Tensor({1});
    return dummy_weights;
  } // Just return one for interface

  void set_weights(const Tensor &weights) override
  {
    // Assuming weights contains both WQ and WK concatenated
    // weights_query = weights.slice(1, 0, weights.size(1));
    // weights_key = weights.slice(1, weights.size(1), 2 * weights.size(1));
  }

  const Tensor &get_deltas() const override
  {
    // Return dummy tensor - would need proper implementation
    static Tensor dummy;
    return dummy;
  }

  int input_size() const override
  {
    // Returns total size of input (Xin + T)
    // return input_feature_map.size(0) * input_feature_map.size(1) +
    //        visual_tokens.size(0) * visual_tokens.size(1);
    return 0;
  }

  int output_size() const override
  {
    // return output.size(0) * output.size(1); // HW × C
    return 0;
  }

  bool has_weights() const override { return true; }

  void set_training(bool training) override {}

  const Tensor &get_input_deltas() const override
  {
    return input_deltas; // [N, HW, C_in]
  }

  Tensor get_token_deltas() const
  {
    return token_deltas; // [N, L, D]
  }
  void apply_gradients(float batch_size) override
  {
    // Escalar los gradientes por el tamaño del batch

    // if (optimizer)
    // {
    //   gradients_key = gradients_key / batch_size;
    //   gradients_query = gradients_query / batch_size;
    //   Tensor a = gradients_key;
    //   Tensor b = gradients_query;
    //   optimizer->update(weights_query, gradients_query, /*bias=*/weights_key, /*bias_grad=*/gradients_key);
    // }
  }
  void accumulate_gradients() override
  {
  }
};