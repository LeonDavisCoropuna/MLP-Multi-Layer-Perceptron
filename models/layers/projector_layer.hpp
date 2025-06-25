#pragma once
#include <vector>
#include <random>
#include "../../utils/tensor.hpp"
#include "layer.hpp"

class ProjectorLayer : public Layer
{
private:
  Tensor weights_query; // WQ ∈ R^{C×C}
  Tensor weights_key;   // WK ∈ R^{C×C}

  Tensor input_feature_map; // Xin ∈ R^{HW×C}
  Tensor visual_tokens;     // T ∈ R^{L×C} (output from transformer)

  Tensor output; // Xout ∈ R^{HW×C}
  Tensor gradients_query;
  Tensor gradients_key;

  bool is_training;
  int num_tokens; // número de tokens L
  int channels;
public:
  ProjectorLayer(int n_channels, int n_tokens) : weights_query(Tensor({n_channels, n_channels})),
                                               weights_key(Tensor({n_channels, n_channels})),
                                               gradients_query(Tensor({n_channels, n_channels})),
                                               gradients_key(Tensor({n_channels, n_channels})),
                                               num_tokens(n_tokens), channels(n_channels)
  {
    // Initialize weights (could use Xavier or other initialization)
    weights_query.rand_init(0.0f, 1.0f / sqrtf(channels));
    weights_key.rand_init(0.0f, 1.0f / sqrtf(channels));
  }

  Tensor forward(const Tensor &input) override
  {
    // Input shape: [B, HW+L, C]
    if (input.shape.size() != 3)
    {
      throw std::runtime_error("ProjectorLayer::forward: Input must be 3D [B, HW+L, C]");
    }

    int B = input.shape[0];
    int total = input.shape[1]; // HW + L
    int C = input.shape[2];

    if (C != channels)
    {
      throw std::runtime_error("Channel dimension mismatch");
    }

    int HW = total - num_tokens;
    if (HW <= 0)
    {
      throw std::runtime_error("HW dimension must be positive");
    }

    // 1. Separar el input concatenado
    input_feature_map = input.slice(1, 0, HW);           // [B, HW, C]
    visual_tokens = input.slice(1, HW, HW + num_tokens); // [B, L, C]

    // 2. Calcular queries y keys
    Tensor queries = input_feature_map.matmul(weights_query); // [B, HW, C]
    Tensor keys = visual_tokens.matmul(weights_key);          // [B, L, C]

    // 3. Calcular atención
    Tensor attention = queries.matmul(keys.transpose({0, 2, 1})); // [B, HW, L]
    attention = attention.softmax(2);                             // Softmax sobre la dimensión L

    // 4. Aplicar atención
    Tensor attended = attention.matmul(visual_tokens); // [B, HW, C]

    // 5. Conexión residual
    output = input_feature_map + attended; // [B, HW, C]

    return output;
  }

  // Placeholder for backward pass - would need to be implemented
  void backward(const Tensor *targets = nullptr,
                const Layer *next_layer = nullptr) override
  {
    // Implementation would compute gradients for WQ and WK
    // Based on chain rule from next layer's deltas
  }

  // Placeholder for update_weights
  void update_weights() override
  {
    // Update weights_query and weights_key using their gradients
  }

  // Placeholder for zero_grad
  void zero_grad() override
  {
    gradients_query.init();
    gradients_key.init();
  }

  const Tensor &get_outputs() const override { return output; }
  const Tensor &get_weights() const override { return weights_query; } // Just return one for interface

  void set_weights(const Tensor &weights) override
  {
    // Assuming weights contains both WQ and WK concatenated
    weights_query = weights.slice(1, 0, weights.size(1));
    weights_key = weights.slice(1, weights.size(1), 2 * weights.size(1));
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
    return input_feature_map.size(0) * input_feature_map.size(1) +
           visual_tokens.size(0) * visual_tokens.size(1);
  }

  int output_size() const override
  {
    return output.size(0) * output.size(1); // HW × C
  }

  bool has_weights() const override { return true; }

  void set_training(bool training) override { is_training = training; }

  const Tensor &get_input_deltas() const override
  {
    // Return dummy tensor - would need proper implementation
    static Tensor dummy;
    return dummy;
  }
  void accumulate_gradients() {}
  void apply_gradients(float batch_size) {}
};