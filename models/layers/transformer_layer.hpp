#pragma once
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
#include "layer.hpp"
#include "../../utils/activations.hpp"

class TransformerLayer : public Layer
{
private:
  Tensor weights_K;  // Key weights (C x C)
  Tensor weights_Q;  // Query weights (C x C)
  Tensor weights_F1; // First feedforward weights (C x C)
  Tensor weights_F2; // Second feedforward weights (C x C)
  Tensor grad_K;
  Tensor grad_Q;
  Tensor grad_F1;
  Tensor grad_F2;

  Tensor weights_V; // Value weights (C x C) <- ¡Nuevo!
  Tensor weights_O; // Proyección final (C x C) <- ¡Nuevo!

  Tensor outputs;
  Tensor input_deltas;
  Tensor deltas;

  int num_heads; // Número de cabezas de atención
  int head_dim;  // Dimensión de cada cabeza (C / num_heads)
  bool training;
  ActivationFunction *relu;

public:
  TransformerLayer(int channels, int heads = 8) : num_heads(heads),
                                                  head_dim(channels / heads),
                                                  weights_K(Tensor({channels, channels})),
                                                  weights_Q(Tensor({channels, channels})),
                                                  weights_V(Tensor({channels, channels})),      // Values
                                                  weights_O(Tensor({channels, channels})),      // Output projection
                                                  weights_F1(Tensor({channels, channels * 4})), // Feedforward expansion
                                                  weights_F2(Tensor({channels * 4, channels}))  // Feedforward contraction
  {
    // Inicialización (Xavier/Kaiming)
    weights_K.rand_init();
    weights_Q.rand_init();
    weights_V.rand_init();
    weights_O.rand_init();
    weights_F1.rand_init();
    weights_F2.rand_init();
  }

  Tensor forward(const Tensor &input) override
  {
    int batch_size = input.shape[0];
    int L = input.shape[1]; // Secuencia length
    int C = input.shape[2]; // Embedding dim

    // 1. Multi-Head Projections
    Tensor K = input.matmul(weights_K).reshape({batch_size, L, num_heads, head_dim});
    Tensor Q = input.matmul(weights_Q).reshape({batch_size, L, num_heads, head_dim});
    Tensor V = input.matmul(weights_V).reshape({batch_size, L, num_heads, head_dim});

    // 2. Scaled Dot-Product Attention (por cabeza)

    Tensor Q_t = Q.transpose({0, 2, 1, 3});                                 // [B, H, L, D]
    Tensor K_t = K.transpose({0, 2, 3, 1});                                 // [B, H, D, L]
    Tensor attention_scores = Q_t.matmul(K_t) / std::sqrt((float)head_dim); // [B, H, L, L]

    attention_scores = attention_scores.softmax(3); // Softmax en la dimensión de keys

    // 3. Aplicar atención a los values
    Tensor V_t = V.transpose({0, 2, 1, 3}); // {B, H, L_k, D}
    Tensor attended = attention_scores.matmul(V_t); // {batch_size, L, num_heads, head_dim}

    // 4. Concatenar y proyectar
    attended = attended.reshape({batch_size, L, C});
    Tensor output = attended.matmul(weights_O); // Proyección final

    // 5. Add & Norm (simplificado)
    output = output + input; // Residual connection

    // 6. Feedforward Network (2 capas)
    Tensor ff = output.matmul(weights_F1).relu();
    ff = ff.matmul(weights_F2);

    return output + ff; // Residual + salida
  }

  // Other required implementations (placeholders for now)
  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    // To be implemented
  }

  void update_weights() override
  {
    // To be implemented
  }

  void zero_grad() override
  {
    // grad_K.zero();
    // grad_Q.zero();
    // grad_F1.zero();
    // grad_F2.zero();
  }

  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_weights() const override { return weights_K; } // Just return one for interface

  void set_weights(const Tensor &w) override
  {
    // Would need to handle all weights properly in real implementation
    weights_K = w;
  }

  const Tensor &get_deltas() const override { return deltas; }
  const Tensor &get_input_deltas() const override { return input_deltas; }

  int input_size() const override { return weights_K.shape[0]; }
  int output_size() const override { return weights_K.shape[1]; }

  bool has_weights() const override { return true; }
  void set_training(bool is_training) override { training = is_training; }

  void accumulate_gradients() {}
  void apply_gradients(float batch_size) {}
};