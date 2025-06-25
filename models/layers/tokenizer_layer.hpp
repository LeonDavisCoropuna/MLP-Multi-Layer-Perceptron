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
  Tensor xin; // Salida Xin para concatenar externamente
  Tensor tokens;

public:
  // Constructor for filter-based tokenizer (first layer)
  TokenizerLayer(int channels, int num_tokens) : num_tokens(num_tokens),
                                                 is_recurrent(false),
                                                 weights_WA(Tensor({channels, num_tokens})),
                                                 weights_WTtoR(Tensor()) // Empty for filter-based
  {
    weights_WA.rand_init();
  }

  // // Constructor for recurrent tokenizer (subsequent layers)
  // TokenizerLayer(int channels, int num_tokens, bool recurrent) : num_tokens(num_tokens),
  //                                                                is_recurrent(recurrent),
  //                                                                weights_WA(Tensor()), // Empty for recurrent
  //                                                                weights_WTtoR(Tensor({channels, channels}))
  // {
  //   weights_WTtoR.rand_init();
  // }

  Tensor forward(const Tensor &input) override
  {
    if (input.shape.size() != 4)
      throw std::invalid_argument("Input must be 4D: {batch, C, H, W}");

    int B = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int HW = H * W;

    // 1. Aplanar el mapa espacial: [B, C, H, W] → [B, HW, C]
    Tensor X = input.reshape({B, HW, C});
    xin = X; // Guardar Xin para uso posterior (por ejemplo, en ProjectorLayer)
    Tensor A;
    if (is_recurrent)
    {
      if (prev_tokens.shape.size() != 3 || weights_WTtoR.shape.size() != 2)
        throw std::runtime_error("prev_tokens or weights_WTtoR has invalid shape");

      Tensor WR = prev_tokens.matmul(weights_WTtoR); // {B, L, C}
      Tensor WR_T = WR.transpose({0, 2, 1});         // {B, C, L}
      A = X.matmul(WR_T);                            // {B, HW, L}
    }
    else
    {
      if (weights_WA.shape.size() != 2)
        throw std::runtime_error("weights_WA must be 2D");

      A = X.matmul(weights_WA); // {B, HW, L}
    }

    // 2. Softmax en la dimensión HW
    A = A.softmax(1);
    attention_weights = A;

    // 3. T = A^T * X → {B, L, C}
    Tensor A_T = A.transpose({0, 2, 1}); // {B, L, HW}
    Tensor T = A_T.matmul(X);            // {B, L, C}
    prev_tokens = T.detach();
    tokens = T;
    
    return tokens;
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

  Tensor get_tokens() const { return tokens; }
  Tensor get_xin() const { return xin; }

  void accumulate_gradients() {}
  void apply_gradients(float batch_size) {}
};