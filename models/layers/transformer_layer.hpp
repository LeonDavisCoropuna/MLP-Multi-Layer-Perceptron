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
  Tensor grad_V;
  Tensor grad_F1;
  Tensor grad_F2;

  // Cache para backward
  Tensor Tin;              // Input original [B, L, C]
  Tensor attention_scores; // [B, H, L, L]
  Tensor ffn_intermediate; // σ(T'_out F1) [B, L, C]
  Tensor T_out_prime;      // Output después de atención [B, L, C]
  Tensor Q, K, V;          // Proyecciones para atención [B, H, L, D]

  Tensor weights_V; // Value weights (C x C) <- ¡Nuevo!
  Tensor weights_O; // Proyección final (C x C) <- ¡Nuevo!

  Tensor outputs;
  Tensor input_deltas;
  Tensor deltas;

  int num_heads; // Número de cabezas de atención
  int head_dim;  // Dimensión de cada cabeza (C / num_heads)
  int B, L, C;
  bool is_training;

  ActivationFunction *activation;
  Optimizer *optimizer;

public:
  TransformerLayer(int channels, int heads, ActivationFunction *activate, Optimizer *optim)
      : num_heads(heads),
        head_dim(channels / heads),
        weights_K(Tensor({channels, channels})),
        weights_Q(Tensor({channels, channels})),
        weights_V(Tensor({channels, channels})),      // Values
        weights_O(Tensor({channels, channels})),      // Output projection
        weights_F1(Tensor({channels, channels * 4})), // Feedforward expansion
        weights_F2(Tensor({channels * 4, channels})), // Feedforward contraction
        activation(activate), optimizer(optim)
  {
    // Inicialización (Xavier/Kaiming)
    weights_K.init_xavier();
    weights_Q.init_xavier();
    weights_V.init_xavier();
    weights_O.init_xavier();
    weights_F1.init_xavier();
    weights_F2.init_xavier();
  }

  Tensor forward(const std::vector<Tensor> &input_) override
  {
    auto input = input_[0];
    Tin = input; // Guardar input original para backward

    // ✅ Asignar a miembros de clase
    B = input.shape[0];
    L = input.shape[1];
    C = input.shape[2];

    // 1. Multi-head projections
    Q = input.matmul(weights_Q).reshape({B, L, num_heads, head_dim});
    K = input.matmul(weights_K).reshape({B, L, num_heads, head_dim});
    V = input.matmul(weights_V).reshape({B, L, num_heads, head_dim});

    // 2. Transpose to [B, H, L, D]
    Q = Q.transpose({0, 2, 1, 3});
    K = K.transpose({0, 2, 1, 3});
    V = V.transpose({0, 2, 1, 3});

    // 3. Attention scores
    Tensor K_t = K.transpose({0, 1, 3, 2});
    attention_scores = Q.matmul(K_t) / std::sqrt((float)head_dim);
    attention_scores = attention_scores.softmax(3); // softmax en la última dimensión (L)

    // 4. Aplicar atención a V
    Tensor attended = attention_scores.matmul(V); // [B, H, L, D]

    // 5. Transponer y unir cabezas: [B, L, C]
    attended = attended.transpose({0, 2, 1, 3}).reshape({B, L, C});

    // 6. Proyección final
    Tensor projected = attended.matmul(weights_O); // [B, L, C]

    // 7. Primera conexión residual
    T_out_prime = projected + input;

    // 8. Feedforward
    ffn_intermediate = T_out_prime.matmul(weights_F1).relu(); // σ(W1)
    Tensor ff = ffn_intermediate.matmul(weights_F2);          // * W2

    // 9. Segunda conexión residual
    Tensor output = T_out_prime + ff;

    return output;
  }

  void backward(const Tensor *next_layer_deltas = nullptr, const Layer *next_layer = nullptr) override
  {
    if (!is_training)
      return;

    // Obtener deltas de la siguiente capa [B=32, L=16, C=8]
    const Tensor &delta_out = (next_layer != nullptr) ? next_layer->get_input_deltas() : *next_layer_deltas;

    int B = delta_out.shape[0]; // 32
    int L = delta_out.shape[1]; // 16
    int C = delta_out.shape[2]; // 8

    // 1. Gradiente de la segunda conexión residual
    Tensor delta_ffn = delta_out; // [32, 16, 8]

    // 2. Backward a través del FFN (Feed Forward Network)
    // Capa F2
    grad_F2 = ffn_intermediate.transpose({0, 2, 1}).matmul(delta_ffn).mean(0); // [C, C]
    Tensor delta_ffn_hidden = delta_ffn.matmul(weights_F2.transpose());        // [32, 16, 8]

    // Gradiente de ReLU
    delta_ffn_hidden = delta_ffn_hidden * (ffn_intermediate.greater_than(0)); // [32, 16, 8]

    // Capa F1
    grad_F1 = T_out_prime.transpose({0, 2, 1}).matmul(delta_ffn_hidden).mean(0); // [C, C]
    Tensor delta_T_out_prime = delta_ffn_hidden.matmul(weights_F1.transpose());  // [32, 16, 8]

    // Sumar gradiente residual
    delta_T_out_prime = delta_T_out_prime + delta_out; // [32, 16, 8]

    // 3. Backward a través de la proyección de salida (weights_O)
    Tensor delta_attended = delta_T_out_prime; // [32, 16, 8]

    // 4. Reorganizar para multi-head attention [B, L, C] -> [B, H, L, D]
    delta_attended = delta_attended.reshape({B, L, num_heads, head_dim}).transpose({0, 2, 1, 3}); // [32, 4, 16, 2]

    // 5. Gradiente de la multiplicación atención-valores
    Tensor delta_attention_scores = delta_attended.matmul(V.transpose({0, 1, 3, 2})); // [32, 4, 16, 16]
    Tensor delta_V = attention_scores.transpose({0, 1, 3, 2}).matmul(delta_attended); // [32, 4, 16, 2]

    // 6. Backward a través del softmax
    Tensor delta_attention_pre_softmax = attention_softmax_backward(delta_attention_scores, attention_scores); // [32, 4, 16, 16]
    delta_attention_pre_softmax = delta_attention_pre_softmax / sqrtf(static_cast<float>(head_dim));

    // 7. Gradientes para Q y K
    Tensor delta_Q = delta_attention_pre_softmax.matmul(K);                         // [32, 4, 16, 2]
    Tensor delta_K = delta_attention_pre_softmax.transpose({0, 1, 3, 2}).matmul(Q); // [32, 4, 16, 2]

    // 8. Reorganizar y calcular gradientes de pesos
    delta_Q = delta_Q.transpose({0, 2, 1, 3}).reshape({B, L, C}); // [32, 16, 8]
    delta_K = delta_K.transpose({0, 2, 1, 3}).reshape({B, L, C}); // [32, 16, 8]
    delta_V = delta_V.transpose({0, 2, 1, 3}).reshape({B, L, C}); // [32, 16, 8]

    grad_Q = Tin.transpose({0, 2, 1}).matmul(delta_Q).mean(0); // [C, C]
    grad_K = Tin.transpose({0, 2, 1}).matmul(delta_K).mean(0); // [C, C]
    grad_V = Tin.transpose({0, 2, 1}).matmul(delta_V).mean(0); // [C, C]

    // 9. Gradiente final para la capa anterior
    input_deltas = delta_Q.matmul(weights_Q.transpose()) +
                   delta_K.matmul(weights_K.transpose()) +
                   delta_V.matmul(weights_V.transpose()) +
                   delta_T_out_prime; // [32, 16, 8]
  }

  Tensor attention_softmax_backward(const Tensor &delta, const Tensor &attention)
  {
    // delta: [B, H, L, L] (gradiente después del softmax)
    // attention: [B, H, L, L] (valores después del softmax)

    Tensor result(delta.shape); // [B, H, L, L]

    for (int b = 0; b < delta.shape[0]; ++b)
    {
      for (int h = 0; h < delta.shape[1]; ++h)
      {
        for (int i = 0; i < delta.shape[2]; ++i)
        {
          // Extraer vectores 1D
          Tensor row = attention.slice(0, b, b + 1)
                           .slice(1, h, h + 1)
                           .slice(2, i, i + 1)
                           .reshape({delta.shape[3]}); // [L]

          Tensor grad_row = delta.slice(0, b, b + 1)
                                .slice(1, h, h + 1)
                                .slice(2, i, i + 1)
                                .reshape({delta.shape[3]}); // [L]

          // Jacobiano del softmax: diag(p) - p*p^T
          Tensor jacobian = row.diag() - row.outer(row); // [L, L]

          // Multiplicar por el gradiente: J^T * grad
          Tensor grad = jacobian.transpose().matmul(grad_row); // [L]

          // Almacenar resultado
          Tensor result_slice = result.slice(0, b, b + 1)
                                    .slice(1, h, h + 1)
                                    .slice(2, i, i + 1)
                                    .reshape(grad.shape);
          result_slice.copy_(grad);
        }
      }
    }
    return result;
  }

  void update_weights() override
  {
    if (!optimizer)
      return;

    Tensor dummy_bias = Tensor(); // forma escalar o vacía
    Tensor dummy_grad = Tensor();
    optimizer->update(weights_Q, grad_Q, dummy_bias, dummy_grad);
    optimizer->update(weights_K, grad_K, dummy_bias, dummy_grad);
    optimizer->update(weights_V, grad_V, dummy_bias, dummy_grad);
    optimizer->update(weights_O, grad_V, dummy_bias, dummy_grad); // Si usas un `grad_O`, cámbialo aquí
    optimizer->update(weights_F1, grad_F1, dummy_bias, dummy_grad);
    optimizer->update(weights_F2, grad_F2, dummy_bias, dummy_grad);
  }
  void zero_grad() override
  {
    grad_K.init();
    grad_Q.init();
    grad_F1.init();
    grad_F2.init();
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
  void set_training(bool is_training_) override { is_training = is_training_; }

  void accumulate_gradients() {}

  void apply_gradients(float batch_size) override
  {

    Tensor dummy_bias; // forma escalar o vacía
    Tensor dummy_grad;

    grad_Q = grad_Q / batch_size;
    grad_K = grad_K / batch_size;
    grad_V = grad_V / batch_size;
    grad_F1 = grad_F1 / batch_size;
    grad_F2 = grad_F2 / batch_size;

    Tensor a = grad_Q;
    Tensor b = grad_K;
    Tensor c = grad_V;
    Tensor d = grad_F1;
    Tensor e = grad_F2;

    optimizer->update(weights_Q, grad_Q, dummy_bias, dummy_grad);
    optimizer->update(weights_K, grad_K, dummy_bias, dummy_grad);
    optimizer->update(weights_V, grad_V, dummy_bias, dummy_grad);
    optimizer->update(weights_O, grad_V, dummy_bias, dummy_grad); // Si usas un `grad_O`, cámbialo aquí
    optimizer->update(weights_F1, grad_F1, dummy_bias, dummy_grad);
    optimizer->update(weights_F2, grad_F2, dummy_bias, dummy_grad);
  }
};