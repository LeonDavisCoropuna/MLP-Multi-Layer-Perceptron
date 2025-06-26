#pragma once
#include "../filter_tokenizer.hpp"
#include "../transformer_layer.hpp"
#include "../projector_layer.hpp"
#include "../../../utils/activations.hpp"
#include "../../../utils/optimizer.hpp"

class VisionTransformerBlock : public Layer
{
private:
  FilterTokenizer *tokenizer;
  TransformerLayer *transformer;
  ProjectorLayer *projector;

  Tensor xin;     // Almacena features originales [B, HW, C]
  Tensor tokens;  // Almacena tokens [B, L, C]
  Tensor outputs; // Salida del bloque [B, C, H, W]
  bool is_training;
  ActivationFunction *activation;
  Optimizer *optimizer;
  Tensor input_deltas;

  std::string tokenizer_type = "filter";
  bool is_projected = true;

public:
  VisionTransformerBlock(int channels, int num_tokens, int num_heads, bool is_recurrent,
                         ActivationFunction *activate, Optimizer *optim)
      : is_training(is_recurrent), activation(activate), optimizer(optim)
  {
    int token_channels = channels; // Puedes usar el mismo número de canales o diferente
    tokenizer = new FilterTokenizer(channels, token_channels, num_tokens);

    // 2. Inicializar el Transformer
    // Asumiendo que tienes una clase TransformerLayer con estos parámetros:
    transformer = new TransformerLayer(token_channels, num_heads, activate, optim);

    // 3. Inicializar el Projector
    // Usamos channels tanto para entrada como salida para mantener dimensiones
    projector = new ProjectorLayer(channels, channels, token_channels);
  }
  ~VisionTransformerBlock() override
  {
    // Limpieza de memoria
    delete tokenizer;
    delete transformer;
    delete projector;
  }

  Tensor forward(const std::vector<Tensor> &input_) override
  {
    auto input = input_[0];
    // Paso 1: Almacenar dimensiones originales y aplanar a [N, HW, C]
    int N = input.shape[0]; // batch
    int C = input.shape[1]; // channels
    int H = input.shape[2]; // height
    int W = input.shape[3]; // width
    int HW = H * W;

    // Convertir [N, C, H, W] -> [N, HW, C] (igual que PyTorch espera)
    Tensor x = input.reshape({N, C, HW}).transpose({0, 2, 1});

    // Paso 2: Aplicar tokenizer (implementación depende de tokenizer_type)
    Tensor t;
    if (tokenizer_type == "filter")
    {
      t = tokenizer->forward({x});
    }
    // else
    // {
    //   // Si tu implementación necesita el tensor temporal t
    //   Tensor temp_t; // Necesitarías obtener este valor de algún lado
    //   t = tokenizer->forward(x, temp_t);
    // }

    // Paso 3: Permutar tokens para transformer [N, L, C] -> [L, N, C]
    Tensor t_transformer = t.transpose({0, 1, 2});

    // Paso 4: Aplicar transformer
    Tensor t_out = transformer->forward({t_transformer});

    // Paso 5: Permutar de vuelta [L, N, C] -> [N, L, C]
    t_out = t_out.transpose({0, 1, 2});
    t = t_transformer.transpose({0, 1, 2}); // También permutar t de vuelta

    // Paso 6: Solo aplicar projector si está configurado
    Tensor out;
    if (is_projected)
    {
      out = projector->forward({x, t_out});
      return out; // Retorna solo el output proyectado (como en PyTorch cuando is_projected=true)
    }

    return t_out; // Retorna los tokens si no está proyectado
  }

  void backward(const Tensor *next_layer_deltas = nullptr, const Layer *next_layer = nullptr) override
  {
    // 1. Obtener los deltas de la siguiente capa [batch, C, H, W]
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

    // 2. Convertir delta_out a la forma [N, HW, C]
    int N = delta_out.shape[0];
    int C = delta_out.shape[1];
    int H = delta_out.shape[2];
    int W = delta_out.shape[3];
    int HW = H * W;

    Tensor delta_out_flat = delta_out.reshape({N, C, HW}).transpose({0, 2, 1});

    // 3. Backward a través del Projector (si está activado)
    Tensor delta_proj, delta_tokens_out;
    if (is_projected)
    {
      projector->backward(&delta_out_flat);

      // Obtener los deltas del projector
      delta_proj = projector->get_input_deltas();       // [N, HW, C]
      delta_tokens_out = projector->get_token_deltas(); // [N, L, C]
    }
    else
    {
      delta_tokens_out = delta_out_flat; // Si no hay projector, los deltas van directo a los tokens
    }

    // 4. Backward a través del Transformer
    // Permutar los deltas de tokens [N, L, C] -> [L, N, C]
    Tensor delta_tokens_trans = delta_tokens_out.transpose({0, 1, 2});
    transformer->backward(&delta_tokens_trans);

    // Obtener deltas del transformer [L, N, C] -> [N, L, C]
    Tensor delta_tokens = transformer->get_input_deltas().transpose({0, 1, 2});

    // 5. Backward a través del Tokenizer
    tokenizer->backward(&delta_tokens);
    Tensor delta_xin = tokenizer->get_input_deltas(); // [N, HW, C]

    // 6. Sumar gradientes si el projector contribuyó (residual connection)
    if (is_projected)
    {
      delta_xin = delta_xin + delta_proj;
    }

    // 7. Convertir los deltas finales a la forma original [N, C, H, W]
    input_deltas = delta_xin.transpose({0, 2, 1}).reshape({N, C, H, W});

    // 8. Actualizar pesos (opcional, podrías hacerlo en un paso separado)
    if (is_training)
    {
      update_weights();
    }
  }

  void update_weights() override
  {
    // Actualizar pesos de todos los componentes
    if (optimizer != nullptr)
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
    return input_deltas; // [batch, C, H, W]
  }

  // Métodos adicionales para acceso a estados internos
  const Tensor &get_tokens() const { return tokens; }
  const Tensor &get_xin() const { return xin; }
  const Tensor &get_attention_weights() const { return tokenizer->get_attention_weights(); }

  const Tensor &get_input_deltas()
  {
    return input_deltas;
  }
};