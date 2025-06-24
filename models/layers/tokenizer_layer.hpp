#pragma once
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
#include "layer.hpp"

class TokenizerLayer : public Layer
{
private:
  int num_tokens; // L (number of visual tokens)
  bool is_recurrent;

  // For filter-based tokenizer
  Tensor weights_WA; // C x L (for initial tokenization)

  // For recurrent tokenizer
  Tensor weights_WTtoR; // C x C (for recurrent tokenization)
  Tensor prev_tokens;   // Stores tokens from previous forward pass

  Tensor attention_weights; // Spatial attention map
  Tensor outputs;

public:
  // Constructor for filter-based tokenizer (first layer)
  TokenizerLayer(int channels, int num_tokens) : num_tokens(num_tokens),
                                                 is_recurrent(false),
                                                 weights_WA(Tensor({channels, num_tokens})),
                                                 weights_WTtoR(Tensor()) // Empty for filter-based
  {
    weights_WA.rand_init();
  }

  // Constructor for recurrent tokenizer (subsequent layers)
  TokenizerLayer(int channels, int num_tokens, bool recurrent) : num_tokens(num_tokens),
                                                                 is_recurrent(true),
                                                                 weights_WA(Tensor()), // Empty for recurrent
                                                                 weights_WTtoR(Tensor({channels, channels}))
  {
    weights_WTtoR.rand_init();
  }

  Tensor forward(const Tensor &input) override
  {
    // Validar que la entrada tenga 4 dimensiones
    if (input.shape.size() != 4)
      throw std::invalid_argument("Input must be 4D: {batch, C, H, W}");

    int batch_size = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int HW = H * W;

    // 1. Reshape: {batch, C, H, W} → {batch, HW, C}
    Tensor X = input.reshape({batch_size, HW, C});

    // 2. Transpose: {batch, HW, C} → {batch, C, HW}
    // X = X.transpose({0, 2, 1}); // valid axis permutation

    Tensor A;

    if (is_recurrent)
    {
      // Recurrent tokenizer (Equation 2)
      if (prev_tokens.shape.size() != 3 || weights_WTtoR.shape.size() != 2)
        throw std::runtime_error("prev_tokens or weights_WTtoR has invalid shape");

      // WR = prev_tokens * WTtoR  → {batch, L, C}
      Tensor WR = prev_tokens.matmul(weights_WTtoR);

      // A = softmax(X * WR^T)     → {batch, HW, L}
      Tensor WR_T = WR.transpose({0, 2, 1});
      A = X.matmul(WR_T);
    }
    else
    {
      // Filter-based tokenizer (Equation 1)
      if (weights_WA.shape.size() != 2)
        throw std::runtime_error("weights_WA must be 2D for filter-based tokenizer");

      // A = softmax(X * WA)       → {batch, HW, L}
      A = X.matmul(weights_WA); // weights_WA: {C, L} //aqui muere
    }

    // 3. Softmax en dimensión HW (dim=1)
    A = A.softmax(1);

    // Guardar pesos de atención
    attention_weights = A;

    // 4. T = A^T * X → {batch, L, C}
    Tensor A_T = A.transpose({0, 2, 1}); // {batch, L, HW}
    Tensor T = A_T.matmul(X);            // {batch, L, C}

    // Guardar para uso recurrente
    // prev_tokens = T.detach();  // Si tienes detach implementado

    outputs = T;
    return outputs;
  }

  // Other required Layer interface methods
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
    // To be implemented
  }

  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_weights() const override
  {
    return is_recurrent ? weights_WTtoR : weights_WA;
  }

  void set_weights(const Tensor &w) override
  {
    if (is_recurrent)
    {
      weights_WTtoR = w;
    }
    else
    {
      weights_WA = w;
    }
  }

  const Tensor &get_deltas() const override
  {
    static Tensor dummy;
    return dummy;
  }

  const Tensor &get_input_deltas() const override
  {
    static Tensor dummy;
    return dummy;
  }

  int input_size() const override
  {
    return weights_WA.shape[0] * weights_WA.shape[1];
  }

  int output_size() const override
  {
    return num_tokens * weights_WA.shape[0];
  }

  bool has_weights() const override { return true; }
  void set_training(bool is_training) override { /* Not needed */ }

  // Additional methods specific to Tokenizer
  const Tensor &get_attention_weights() const { return attention_weights; }
  int get_num_tokens() const { return num_tokens; }
  void accumulate_gradients() {}
  void apply_gradients(float batch_size) {}
};