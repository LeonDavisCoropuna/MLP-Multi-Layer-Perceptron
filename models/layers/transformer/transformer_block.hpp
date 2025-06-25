#pragma once
#include "../tokenizer_layer.hpp"
#include "../transformer_layer.hpp"
#include "../projector_layer.hpp"

class VisionTransformerBlock : public Layer
{
private:
  TokenizerLayer *tokenizer;
  TransformerLayer *transformer;
  ProjectorLayer *projector;

  Tensor xin;     // Almacena features originales [B, HW, C]
  Tensor tokens;  // Almacena tokens [B, L, C]
  Tensor outputs; // Salida del bloque [B, C, H, W]
  bool is_training;

public:
  VisionTransformerBlock(int channels, int num_tokens, int num_heads, bool is_recurrent = false)
      : is_training(false)
  {
    // Inicialización con new
    tokenizer = new TokenizerLayer(channels, num_tokens);
    transformer = new TransformerLayer(channels, num_heads);
    projector = new ProjectorLayer(channels, num_tokens);
  }

  ~VisionTransformerBlock() override
  {
    // Limpieza de memoria
    delete tokenizer;
    delete transformer;
    delete projector;
  }

  Tensor forward(const Tensor &input) override
  {
    // entrada es [32,8,13,13]
    // 1. Almacenar features originales [B, C, H, W] -> [B, HW, C]
    int HW = input.shape[2] * input.shape[3];

    // 2. Tokenización
    tokens = tokenizer->forward(input); // [B, L, C]
    xin = input.reshape({input.shape[0], input.shape[1], HW}).transpose({0, 2, 1});

    // 3. Transformer
    Tensor transformed_tokens = transformer->forward(tokens); // [B, L, C]

    // 4. Concatenar Xin (features) y tokens a lo largo de la dimensión HW
    Tensor concatenated = Tensor::concat(xin, transformed_tokens, 1); // [B, HW+L, C]

    // 5. Proyección
    outputs = projector->forward(concatenated); // [B, HW, C] -> luego reshape a [B, C, H, W]
    Tensor outputt = outputs.reshape({input.shape[0], input.shape[1], input.shape[2], input.shape[3]});
    return outputt;
  }

  // Métodos requeridos por la interfaz Layer (stubs por ahora)
  void backward(const Tensor *targets = nullptr, const Layer *next_layer = nullptr) override
  {
    throw std::runtime_error("Backward not implemented yet");
  }

  void update_weights() override
  {
    if (is_training)
    {
      tokenizer->update_weights();
      transformer->update_weights();
      projector->update_weights();
    }
  }

  void accumulate_gradients() override
  {
    tokenizer->accumulate_gradients();
    transformer->accumulate_gradients();
    projector->accumulate_gradients();
  }

  void apply_gradients(float batch_size) override
  {
    tokenizer->apply_gradients(batch_size);
    transformer->apply_gradients(batch_size);
    projector->apply_gradients(batch_size);
  }

  void zero_grad() override
  {
    tokenizer->zero_grad();
    transformer->zero_grad();
    projector->zero_grad();
  }

  const Tensor &get_outputs() const override { return outputs; }
  const Tensor &get_weights() const override { return transformer->get_weights(); }

  void set_weights(const Tensor &weights) override
  {
    // Asume que weights contiene todos los pesos concatenados
    // En una implementación real deberías separarlos para cada subcapa
    transformer->set_weights(weights);
  }

  const Tensor &get_deltas() const override
  {
    static Tensor dummy;
    return dummy;
  }

  int input_size() const override { return tokenizer->input_size(); }
  int output_size() const override { return projector->output_size(); }

  void set_training(bool is_training) override
  {
    this->is_training = is_training;
    tokenizer->set_training(is_training);
    transformer->set_training(is_training);
    projector->set_training(is_training);
  }

  const Tensor &get_input_deltas() const override
  {
    static Tensor dummy;
    return dummy;
  }

  // Métodos adicionales para acceso a estados internos
  const Tensor &get_tokens() const { return tokens; }
  const Tensor &get_xin() const { return xin; }
  const Tensor &get_attention_weights() const { return tokenizer->get_attention_weights(); }
};