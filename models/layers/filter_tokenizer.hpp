#pragma once
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
#include "layer.hpp"

class FilterTokenizer : public Layer
{
private:
  int num_tokens;     // L (number of visual tokens)
  int in_channels;    // C (input channels)
  int token_channels; // D (token channels)

  // Capas lineales equivalentes
  Tensor linear1_weights; // (C, L) - equivalente a nn.Linear(in_channels, tokens)
  Tensor linear2_weights; // (C, D) - equivalente a nn.Linear(in_channels, token_channels)

  // Caches para almacenar valores intermedios (opcional, para debugging)
  Tensor cache1;      // Almacena salida de linear1
  Tensor cache2;      // Almacena softmax
  Tensor token_cache; // Almacena tokens finales

  Tensor output;

public:
  // Constructor for filter-based tokenizer (first layer)
  FilterTokenizer(int in_channels, int token_channels, int num_tokens)
      : in_channels(in_channels),
        token_channels(token_channels),
        num_tokens(num_tokens)
  {

    // Inicialización de pesos (equivalente a xavier_normal_)
    linear1_weights = Tensor({in_channels, num_tokens});
    linear2_weights = Tensor({in_channels, token_channels});

    // Inicialización Xavier normal
    linear1_weights.init_xavier();
    linear2_weights.init_xavier();
  }

  Tensor forward(const std::vector<Tensor> &input_) override
  {
    auto x = input_[0];
    // Verificar dimensiones de entrada [N, HW, C]
    if (x.shape.size() != 3)
    {
      throw std::invalid_argument("Input must be 3D: {batch, HW, C}");
    }

    int N = x.shape[0];
    int HW = x.shape[1];
    int C = x.shape[2];

    // Paso 1: Primera capa lineal (N, HW, C) * (C, L) -> (N, HW, L)
    Tensor a = x.matmul(linear1_weights);
    cache1 = a; // Almacenar para posible uso posterior

    // Paso 2: Softmax sobre la dimensión HW (axis=1)
    a = a.softmax(1);
    cache2 = a;

    // Paso 3: Transponer a (N, L, HW)
    a = a.transpose({0, 2, 1});

    // Paso 4: Multiplicar con entrada original (N, L, HW) * (N, HW, C) -> (N, L, C)
    a = a.matmul(x);

    // Paso 5: Segunda capa lineal (N, L, C) * (C, D) -> (N, L, D)
    Tensor tokens = a.matmul(linear2_weights);
    token_cache = tokens;
    output = tokens;
    return tokens;
  }

  // Other required Layer interface methods
  void backward(const Tensor *next_layer_deltas = nullptr,
                const Layer *next_layer = nullptr) override
  {
    // // if (!is_training)
    // //   return;

    // // Obtener los deltas de la siguiente capa [B=32, L=16, C=8]
    // const Tensor &delta_T = (next_layer != nullptr) ? next_layer->get_input_deltas() : *next_layer_deltas;

    // int B = delta_T.shape[0];            // 32
    // int L = delta_T.shape[1];            // 16
    // int C = delta_T.shape[2];            // 8
    // int HW = attention_weights.shape[1]; // 169

    // // 1. Backward a través de T = A^T * X
    // // delta_T: [32,16,8], xin: [32,169,8]
    // Tensor delta_A_T = delta_T.matmul(xin.transpose({0, 2, 1})); // [32,16,169]

    // // [32,16,169] -> [32,169,16] (transposición correcta)
    // delta_A_T = delta_A_T.transpose({0, 2, 1});

    // // 2. Backward a través de A^T (transposición)
    // Tensor delta_A = delta_A_T; // [32,169,16]

    // // 3. Backward a través del softmax
    // Tensor delta_A_pre_softmax = attention_softmax_backward(delta_A, attention_weights);

    // Tensor delta_X_from_A;
    // Tensor delta_weights;
    // Tensor delta_prev_tokens;

    // if (is_recurrent)
    // {
    //   // 4a. Backward para versión recurrente
    //   Tensor delta_WR = delta_A_pre_softmax.transpose({0, 2, 1}); // [32,16,169]

    //   // Gradientes para weights_WTtoR
    //   gradients_WTtoR = prev_tokens.transpose({0, 2, 1}).matmul(delta_WR).mean(0); // [C,C]

    //   // Gradientes para prev_tokens
    //   delta_prev_tokens = delta_WR.matmul(weights_WTtoR.transpose()); // [32,16,C]

    //   // Backward a través de X * WR^T
    //   delta_X_from_A = delta_A_pre_softmax.matmul(weights_WTtoR.matmul(prev_tokens.transpose({0, 2, 1})));
    // }
    // else
    // {
    //   // 4b. Backward para versión filter-based
    //   gradients_WA = xin.transpose({0, 2, 1}).matmul(delta_A_pre_softmax).mean(0); // [C,L]
    //   delta_X_from_A = delta_A_pre_softmax.matmul(weights_WA.transpose());         // [32,169,C]
    // }

    // // 5. Gradiente directo de X (a través de T = A^T * X)
    // Tensor delta_X_from_T = attention_weights.matmul(delta_T); // [32,169,16] x [32,16,8] = [32,169,8]

    // // 6. Combinar los deltas
    // Tensor delta_X = delta_X_from_T + delta_X_from_A; // [32,169,8]

    // // 7. Reshape de vuelta a [B,C,H,W]
    // input_deltas = delta_X.transpose({0, 2, 1}).reshape({B, C, input_shape[2], input_shape[3]});
  }

  Tensor attention_softmax_backward(const Tensor &delta, const Tensor &attention)
  {
    // delta: [B, HW, L] (gradiente después del softmax)
    // attention: [B, HW, L] (valores después del softmax)

    Tensor result(delta.shape);

    for (int b = 0; b < delta.shape[0]; ++b)
    {
      // Slice para el batch actual
      Tensor attention_batch = attention.slice(0, b, b + 1); // [1, HW, L]
      Tensor delta_batch = delta.slice(0, b, b + 1);         // [1, HW, L]

      for (int hw = 0; hw < delta.shape[1]; ++hw)
      {
        // Slice para la posición HW actual
        Tensor row = attention_batch.slice(1, hw, hw + 1).squeeze();  // [L]
        Tensor grad_row = delta_batch.slice(1, hw, hw + 1).squeeze(); // [L]

        // Jacobiano del softmax: diag(p) - p*p^T
        Tensor jacobian = row.diag() - row.outer(row);
        Tensor grad = grad_row.matmul(jacobian);

        // Insertar el gradiente en la posición correcta
        Tensor result_slice = result.slice(0, b, b + 1).slice(1, hw, hw + 1);
        result_slice.reshape(grad.shape).copy_(grad);
      }
    }
    return result;
  }

  void apply_gradients(float batch_size) override
  {
    // Tensor dummy_bias = Tensor();
    // Tensor dummy_grad = Tensor();

    // if (optimizer)
    // {
    //   if (is_recurrent)
    //   {
    //     gradients_WTtoR = gradients_WTtoR / batch_size;
    //     optimizer->update(weights_WTtoR, gradients_WTtoR, dummy_bias, dummy_grad);
    //   }
    //   else
    //   {
    //     gradients_WA = gradients_WA / batch_size;
    //     Tensor a = gradients_WA;
    //     optimizer->update(weights_WA, gradients_WA, dummy_bias, dummy_grad);
    //   }
    // }
  }

  void zero_grad() override
  {
  //   gradients_WA.init();
  //   gradients_WA.init();
  }

  const Tensor &get_outputs() const override { return output; }
  const Tensor &get_weights() const override
  {
    // return is_recurrent ? weights_WTtoR : weights_WA;
    static Tensor dummy_weights = Tensor({1});
    return dummy_weights;
  }

  void set_weights(const Tensor &w) override
  {
    // if (is_recurrent)
    // {
    //   weights_WTtoR = w;
    // }
    // else
    // {
    //   weights_WA = w;
    // }
  }

  const Tensor &get_deltas() const override
  {
    static Tensor dummy;
    return dummy;
  }

  const Tensor &get_input_deltas() const override
  {
     return output;
  }

  int input_size() const override
  {
     return 0;
  }

  int output_size() const override
  {
     return 0;
  }

  bool has_weights() const override { return true; }
  void set_training(bool is_training) override { /* Not needed */ }

  // Additional methods specific to Tokenizer
  const Tensor &get_attention_weights() const { return output; }
  int get_num_tokens() const { return num_tokens; }

  Tensor get_tokens() const { return output; }
  Tensor get_xin() const { return output; }

  void accumulate_gradients() override {}
  void update_weights() override {}
};