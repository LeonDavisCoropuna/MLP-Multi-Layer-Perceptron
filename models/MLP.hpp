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
      return out[0];
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

  void train(int num_epochs, const vector<vector<float>> &X, const vector<float> &Y, int batch_size = 1, const std::string &log_filepath = "")
  {
    bool is_binary = (layers.back()->list_perceptrons.size() == 1);

    std::ofstream log_file;
    if (!log_filepath.empty())
    {
      log_file.open(log_filepath, std::ios::out); // o std::ios::app si deseas agregar
      if (!log_file.is_open())
      {
        std::cerr << "Error al abrir el archivo de logs: " << log_filepath << std::endl;
        return;
      }
    }

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
      float total_loss = 0.0f;
      int correct_predictions = 0;

      // Procesar en batches
      for (size_t batch_start = 0; batch_start < X.size(); batch_start += batch_size)
      {
        size_t batch_end = std::min(batch_start + batch_size, X.size());
        size_t actual_batch_size = batch_end - batch_start;

        // Acumuladores para gradientes
        vector<vector<vector<float>>> batch_gradients_weights(layers.size());
        vector<vector<float>> batch_gradients_bias(layers.size());

        // Inicializar acumuladores
        for (size_t l = 0; l < layers.size(); l++)
        {
          batch_gradients_weights[l].resize(layers[l]->list_perceptrons.size());
          batch_gradients_bias[l].resize(layers[l]->list_perceptrons.size(), 0.0f);

          for (size_t n = 0; n < layers[l]->list_perceptrons.size(); n++)
          {
            batch_gradients_weights[l][n].resize(
                layers[l]->list_perceptrons[n]->weights.size(), 0.0f);
          }
        }

        float batch_loss = 0.0f;

        // Procesar cada muestra en el batch
        for (size_t i = batch_start; i < batch_end; i++)
        {
          vector<float> outputs = forward(X[i]);
          float y_true = Y[i];

          int predicted_class;
          if (is_binary)
          {
            predicted_class = (outputs[0] > 0.5f) ? 1 : 0;
            if (predicted_class == static_cast<int>(y_true))
              correct_predictions++;
          }
          else
          {
            predicted_class = static_cast<int>(std::distance(
                outputs.begin(),
                std::max_element(outputs.begin(), outputs.end())));
            if (predicted_class == static_cast<int>(y_true))
              correct_predictions++;
          }

          vector<float> target_vec;
          if (is_binary)
          {
            target_vec = {y_true};
          }
          else
          {
            target_vec.assign(layers.back()->list_perceptrons.size(), 0.0f);
            target_vec[static_cast<int>(y_true)] = 1.0f;
          }

          batch_loss += loss_function->compute(outputs, target_vec);

          layers.back()->backward_output_layer(target_vec);
          for (int l = layers.size() - 2; l >= 0; l--)
          {
            layers[l]->backward_hidden_layer(layers[l + 1]);
          }

          for (size_t l = 0; l < layers.size(); l++)
          {
            for (size_t n = 0; n < layers[l]->list_perceptrons.size(); n++)
            {
              auto &neuron = layers[l]->list_perceptrons[n];
              batch_gradients_bias[l][n] += neuron->get_delta();

              for (size_t w = 0; w < neuron->weights.size(); w++)
              {
                batch_gradients_weights[l][n][w] +=
                    neuron->get_delta() * layers[l]->inputs_layer[w];
              }
            }
          }
        }

        for (size_t l = 0; l < layers.size(); l++)
        {
          for (size_t n = 0; n < layers[l]->list_perceptrons.size(); n++)
          {
            auto &neuron = layers[l]->list_perceptrons[n];
            float scale = 1.0f / actual_batch_size;

            for (size_t w = 0; w < neuron->weights.size(); w++)
            {
              batch_gradients_weights[l][n][w] *= scale;
            }
            batch_gradients_bias[l][n] *= scale;

            optimizer->update(neuron->weights,
                              batch_gradients_weights[l][n],
                              neuron->bias,
                              batch_gradients_bias[l][n]);
          }
        }

        total_loss += batch_loss;
        optimizer->increment_t();
      }

      float avg_loss = total_loss / X.size();
      float accuracy = static_cast<float>(correct_predictions) / X.size() * 100.0f;

      std::ostringstream log_stream;
      log_stream << "Epoch " << epoch + 1
                 << ", Loss: " << avg_loss
                 << ", Accuracy: " << accuracy << "%" << std::endl;
      std::cout << "Epoch " << epoch + 1
                 << ", Loss: " << avg_loss
                 << ", Accuracy: " << accuracy << "%" << std::endl;
      if (log_file.is_open())
      {
        log_file << log_stream.str();
      }
      else
      {
        std::cout << log_stream.str();
      }
    }

    if (log_file.is_open())
      log_file.close();
  }
  float evaluate(const vector<vector<float>> &X_test, const vector<float> &Y_test)
  {
    int correct_predictions = 0;

    for (size_t i = 0; i < X_test.size(); i++)
    {
      vector<float> out = forward(X_test[i]);
      int predicted_class;
      float true_class = Y_test[i];

      if (out.size() == 1)
      { // Binario
        predicted_class = out[0] > 0.5f ? 1 : 0;

        if (predicted_class == static_cast<int>(true_class))
        {
          correct_predictions++;
        }
        else
        {
          // std::cerr << "[Error binario] Index: " << i
          //           << ", Predicho: " << predicted_class
          //           << ", Verdadero: " << static_cast<int>(true_class)
          //           << std::endl;
        }
      }
      else
      { // Multiclase
        predicted_class = static_cast<int>(std::distance(out.begin(), std::max_element(out.begin(), out.end())));

        if (predicted_class == static_cast<int>(true_class))
        {
          correct_predictions++;
        }
        else
        {
          // std::cerr << "[Error multiclase] Index: " << i
          //           << ", Predicho: " << predicted_class
          //           << ", Verdadero: " << static_cast<int>(true_class)
          //           << std::endl;
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

  void save_model_weights(const std::string &filename)
  {
    std::ofstream out(filename);
    if (!out.is_open())
    {
      std::cerr << "No se pudo abrir el archivo para guardar el modelo." << std::endl;
      return;
    }

    out << "MLP Model Weights\n";
    out << "Learning Rate: " << learning_rate << "\n";
    out << "Num Layers: " << num_layers.size() << "\n";

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
    {
      const auto *layer = layers[layer_idx];
      out << "Layer " << layer_idx + 1 << "\n";
      out << " - Neurons: " << layer->list_perceptrons.size() << "\n";
      out << " - Learning Rate: " << layer->learning_rate << "\n";

      for (size_t p_idx = 0; p_idx < layer->list_perceptrons.size(); ++p_idx)
      {
        const auto *p = layer->list_perceptrons[p_idx];
        out << "  Neuron " << p_idx + 1 << "\n";
        out << "   - Bias: " << p->bias << "\n";
        out << "   - Weights: ";
        for (float w : p->weights)
        {
          out << w << " ";
        }
        out << "\n";
      }
    }

    out.close();
    std::cout << "Pesos del modelo guardados en: " << filename << std::endl;
  }
  void load_model_weights(const std::string &filename)
  {
    std::ifstream in(filename);
    if (!in.is_open())
    {
      std::cerr << "No se pudo abrir el archivo para cargar el modelo." << std::endl;
      return;
    }

    std::string line;

    // 1) Saltar la cabecera
    std::getline(in, line); // "MLP Model Weights"

    // 2) Leer learning_rate global (opcional usar o ignorar si ya está en memoria)
    std::getline(in, line);
    {
      std::istringstream ss(line);
      std::string tmp;
      ss >> tmp >> tmp;    // "Learning" "Rate:"
      ss >> learning_rate; // valor
    }

    // 3) Leer número de capas (para validación)
    std::getline(in, line);
    {
      std::istringstream ss(line);
      std::string tmp;
      int file_num_layers;
      ss >> tmp >> tmp >> file_num_layers; // "Num" "Layers:" N
      if (file_num_layers != static_cast<int>(layers.size()))
      {
        std::cerr << "Advertencia: número de capas en el archivo ("
                  << file_num_layers << ") no coincide con el modelo ("
                  << layers.size() << ")." << std::endl;
      }
    }

    // 4) Iterar por cada capa
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
    {
      // Leer "Layer N"
      std::getline(in, line);

      // Leer "- Neurons: M" (validar)
      std::getline(in, line);
      int file_neurons = 0;
      {
        std::istringstream ss(line);
        std::string tmp;
        ss >> tmp >> tmp >> file_neurons; // "-" "Neurons:" M
        if (file_neurons != static_cast<int>(layers[layer_idx]->list_perceptrons.size()))
        {
          std::cerr << "Advertencia: neuronas en capa " << layer_idx + 1
                    << " en archivo (" << file_neurons << ") difiere de modelo ("
                    << layers[layer_idx]->list_perceptrons.size() << ")." << std::endl;
        }
      }

      // Leer "- Learning Rate: lr_layer"
      std::getline(in, line);
      {
        std::istringstream ss(line);
        std::string tmp;
        float lr_layer;
        ss >> tmp >> tmp >> lr_layer; // "-" "Learning" "Rate:" lr
        layers[layer_idx]->learning_rate = lr_layer;
      }

      // 5) Iterar perceptrones de la capa
      for (size_t p_idx = 0; p_idx < layers[layer_idx]->list_perceptrons.size(); ++p_idx)
      {
        // Leer "Neuron K"
        std::getline(in, line);

        // Leer " - Bias: value"
        std::getline(in, line);
        {
          std::istringstream ss(line);
          std::string tmp;
          float bias_val;
          ss >> tmp >> tmp >> bias_val; // "-" "Bias:" val
          layers[layer_idx]->list_perceptrons[p_idx]->bias = bias_val;
        }

        // Leer " - Weights: w1 w2 w3 ..."
        std::getline(in, line);
        {
          std::istringstream ss(line);
          std::string tmp;
          ss >> tmp >> tmp; // "-" "Weights:"
          std::vector<float> wts;
          float w;
          while (ss >> w)
          {
            wts.push_back(w);
          }
          layers[layer_idx]->list_perceptrons[p_idx]->weights = std::move(wts);
        }
      }
    }

    in.close();
    std::cout << "Pesos del modelo cargados desde: " << filename << std::endl;
  }
};
