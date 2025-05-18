#include "singleLayerPerceptron.hpp"
#include "../utils/loss.hpp"
class MLP
{
private:
  float learning_rate;
  vector<int> num_layers;
  int num_inputs;
  vector<SingleLayerPerceptron *> layers;
  vector<vector<float>> output_layers;
  vector<ActivationFunction *> activations;
  Loss *loss_function;
  int last_output_size = -1;
  Optimizer *optimizer;

public:
  MLP(float _learning_rate, vector<int> _num_layers,
      vector<ActivationFunction *> _activations, Loss *_loss_function)
  {
    learning_rate = _learning_rate;
    num_layers = _num_layers;
    num_inputs = num_layers[0];
    activations = _activations;
    loss_function = _loss_function;
    int input_size = num_inputs;
    for (size_t i = 0; i < num_layers.size(); i++)
    {
      SingleLayerPerceptron *layer = new SingleLayerPerceptron(num_layers[i], input_size, activations[i], learning_rate, optimizer);
      layers.push_back(layer);
      input_size = num_layers[i];
    }
  }
  MLP(float _learning_rate, Optimizer *_optimizer)
  {
    learning_rate = _learning_rate;
    optimizer = _optimizer;
  }

  void add_layer(int num_neurons, ActivationFunction *activationFunction)
  {
    if (last_output_size == -1)
    {
      throw std::logic_error("Debes añadir una capa de entrada primero o especificar el tamaño inicial");
    }
    SingleLayerPerceptron *layer = new SingleLayerPerceptron(num_neurons, last_output_size, activationFunction, learning_rate, optimizer);
    layers.push_back(layer);
    last_output_size = num_neurons;
  }

  void add_input_layer(int input_size, int num_neurons, ActivationFunction *activationFunction)
  {
    SingleLayerPerceptron *layer = new SingleLayerPerceptron(num_neurons, input_size, activationFunction, learning_rate, optimizer);
    layers.push_back(layer);
    last_output_size = num_neurons;
  }
  void set_loss(Loss *_loss_function)
  {
    loss_function = _loss_function;
  }
  int predict(const vector<float> &input)
  {
    vector<float> out = forward(input);
    if (out.size() == 1)
    { // Caso binario
      return out[0] > 0.5f ? 1 : 0;
    }
    else
    { // Caso multiclase
      return static_cast<int>(std::distance(out.begin(),
                                            std::max_element(out.begin(), out.end())));
    }
  }

  vector<float> forward(vector<float> batch_inputs)
  {
    output_layers.clear();
    vector<float> current_input = batch_inputs;
    for (auto &layer : layers)
    {
      current_input = layer->forward(current_input);
      output_layers.push_back(current_input);
    }
    return current_input;
  }

  void train(int num_epochs, const vector<vector<float>> &X, const vector<float> &Y)
  {
    bool is_binary = (layers.back()->list_perceptrons.size() == 1); // Verifica si es binaria

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
      float total_loss = 0.0f;
      int correct_predictions = 0;

      for (size_t i = 0; i < X.size(); i++)
      {
        vector<float> outputs = forward(X[i]);
        float y_true = Y[i];

        // Manejo de predicciones según tipo de problema
        int predicted_class;
        if (is_binary)
        {
          // Clasificación binaria: umbral 0.5
          predicted_class = (outputs[0] > 0.5f) ? 1 : 0;
        }
        else
        {
          predicted_class = static_cast<int>(std::distance(
              outputs.begin(),
              std::max_element(outputs.begin(), outputs.end())));
        }

        if (is_binary)
        {
          if (predicted_class == static_cast<int>(y_true))
          {
            correct_predictions++;
          }
        }
        else
        {
          if (predicted_class == static_cast<int>(y_true))
          {
            correct_predictions++;
          }
        }

        // Preparar target vector según el tipo de problema
        vector<float> target_vec;
        if (is_binary)
        {
          target_vec = {y_true}; // Solo un valor para BCELoss
        }
        else
        {
          target_vec.assign(layers.back()->list_perceptrons.size(), 0.0f);
          target_vec[static_cast<int>(y_true)] = 1.0f; // One-hot encoding
        }

        // Cálculo de pérdida
        total_loss += loss_function->compute(outputs, target_vec);

        // Backpropagation
        layers.back()->backward_output_layer(target_vec);
        for (int l = layers.size() - 2; l >= 0; l--)
        {
          layers[l]->backward_hidden_layer(layers[l + 1]);
        }
        for (auto &layer : layers)
        {
          layer->update_weights();
        }
      }

      // Cálculo de métricas
      float avg_loss = total_loss / X.size();
      float accuracy = static_cast<float>(correct_predictions) / X.size() * 100.0f;

      std::cout << "Epoch " << epoch + 1
                << ", Loss: " << avg_loss
                << ", Accuracy: " << accuracy << "%" << std::endl;
    }
  }
  float evaluate(const vector<vector<float>> &X_test, const vector<float> &Y_test)
  {
    int correct_predictions = 0;

    for (size_t i = 0; i < X_test.size(); i++)
    {
      vector<float> out = forward(X_test[i]);
      int predicted_class;
      float true_class = Y_test[i]; // Usar float para binario

      if (out.size() == 1)
      { // Binario
        predicted_class = out[0] > 0.5f ? 1 : 0;
        if (predicted_class == static_cast<int>(true_class))
        {
          correct_predictions++;
        }
      }
      else
      { // Multiclase
        predicted_class = static_cast<int>(std::distance(out.begin(),
                                                         std::max_element(out.begin(), out.end())));
        if (predicted_class == static_cast<int>(true_class))
        {
          correct_predictions++;
        }
      }
    }

    float accuracy = static_cast<float>(correct_predictions) / X_test.size() * 100.0f;
    std::cout << "Evaluation Results:" << std::endl;
    std::cout << " - Test samples: " << X_test.size() << std::endl;
    std::cout << " - Correct predictions: " << correct_predictions << std::endl;
    std::cout << " - Accuracy: " << accuracy << "%" << std::endl;

    return accuracy;
  }

  void save_model(const std::string &filename) const
  {
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
      throw std::runtime_error("No se pudo abrir el archivo para guardar el modelo");
    }

    file.write(reinterpret_cast<const char *>(&learning_rate), sizeof(learning_rate));
    size_t num_layers_size = num_layers.size();
    file.write(reinterpret_cast<const char *>(&num_layers_size), sizeof(num_layers_size));
    file.write(reinterpret_cast<const char *>(num_layers.data()),
               num_layers_size * sizeof(int));

    for (const auto &layer : layers)
    {
      layer->serialize(file);
    }
  }

  void load_model(const std::string &filename)
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
      throw std::runtime_error("No se pudo abrir el archivo del modelo: " + filename);
    }

    // Limpiar capas existentes
    for (auto *layer : layers)
    {
      delete layer;
    }
    layers.clear();

    // Leer metadatos
    file.read(reinterpret_cast<char *>(&learning_rate), sizeof(learning_rate));

    size_t num_layers;
    file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

    // Reconstruir arquitectura
    for (size_t i = 0; i < num_layers; ++i)
    {
      // Leer configuración de cada capa
      int neurons, inputs;
      file.read(reinterpret_cast<char *>(&neurons), sizeof(neurons));
      file.read(reinterpret_cast<char *>(&inputs), sizeof(inputs));

      ActivationFunction *activation = new Tanh(); // Temporal - deberías guardar/leer el tipo real
      if (i == num_layers - 1)
      { // Última capa
        activation = new Sigmoid();
      }

      SingleLayerPerceptron *layer = new SingleLayerPerceptron(
          neurons, inputs, activation, learning_rate, optimizer);

      layer->deserialize(file);
      layers.push_back(layer);
    }
  }

  ~MLP()
  {
    for (auto *layer : layers)
    {
      delete layer;
    }
    for (auto *act : activations)
    {
      delete act;
    }
    delete loss_function;
  }
};
