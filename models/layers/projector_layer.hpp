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

public:
  ProjectorLayer(int channels) : weights_query(Tensor({channels, channels})),
                                 weights_key(Tensor({channels, channels})),
                                 gradients_query(Tensor({channels, channels})),
                                 gradients_key(Tensor({channels, channels}))
  {
    // Initialize weights (could use Xavier or other initialization)
    weights_query.rand_init(0.0f, 1.0f / sqrtf(channels));
    weights_key.rand_init(0.0f, 1.0f / sqrtf(channels));
  }

  Tensor forward(const Tensor &input) override
  {
    // Input should be a concatenation of [Xin, T]
    // Assuming input is {HW + L, C} where first HW are Xin, last L are T
    int HW = input.size(0) - input.size(1);                 // L is input.size(1)
    input_feature_map = input.slice(0, 0, HW);              // Xin ∈ R^{HW×C}
    visual_tokens = input.slice(0, HW, HW + input.size(1)); // T ∈ R^{L×C}

    // Compute queries and keys
    Tensor queries = input_feature_map.matmul(weights_query); // (HW×C) @ (C×C) = HW×C

    Tensor keys = visual_tokens.matmul(weights_key);          // (L×C) @ (C×C) = L×C

    // Compute attention weights

    Tensor attention = queries.matmul(keys.transpose()); // (HW×C) @ (C×L) = HW×L
    attention.softmax(1);                                // SOFTMAX over L dimension

    // Project tokens back to feature map space
    output = input_feature_map + attention.matmul(visual_tokens); // (HW×L) @ (L×C) = HW×C

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