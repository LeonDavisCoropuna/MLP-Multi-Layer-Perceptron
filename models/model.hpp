#pragma once
#include <iostream>
#include "layers/layer.hpp"
#include "../utils/loss.hpp"
#include "layers/dense_layer.hpp"
#include "layers/dropout_layer.hpp"
#include "layers/conv2d_layer.hpp"
#include "layers/pooling_layer.hpp"
#include "layers/flatten_layer.hpp"
#include "fstream"
#include "sstream"
class Model
{
private:
  float learning_rate;
  std::vector<int> num_layers;
  int num_inputs;
  std::vector<Tensor> output_layers;
  std::vector<ActivationFunction *> activations;
  Loss *loss_function;
  int last_output_size = -1;
  Optimizer *optimizer;

public:
  std::vector<Layer *> layers;
  Model(float _learning_rate, Optimizer *_optimizer)
  {
    learning_rate = _learning_rate;
    optimizer = _optimizer;
  }

  void add_layer(Layer *layer)
  {
    layers.push_back(layer);
  }

  void set_loss(Loss *_loss_function)
  {
    loss_function = _loss_function;
  }
  int predict(const Tensor &input)
  {
    Tensor out = forward(input);
    if (out.size() == 1)
    { // Caso binario
      return out.data[0];
    }
    else
    { // Caso multiclase
      return static_cast<int>(std::distance(out.data.begin(),
                                            std::max_element(out.data.begin(), out.data.end())));
    }
  }

  Tensor forward(const Tensor &input)
  {
    output_layers.clear();
    Tensor current_input = input;

    for (auto &layer : layers)
    {
      current_input = layer->forward(current_input);
      output_layers.push_back(current_input);
    }

    return current_input;
  }

  virtual void backward_pass(const Tensor &target)
  {
    layers.back()->backward(&target);
    for (int l = layers.size() - 2; l >= 0; l--)
    {
      layers[l]->backward(nullptr, layers[l + 1]);
    }
  }

  // Función para aplicar gradientes
  virtual void apply_gradients(size_t batch_size)
  {
    for (auto &layer : layers)
    {
      layer->apply_gradients(batch_size);
    }
  }

  // ~MLP()
  // {
  //   for (auto *layer : layers)
  //   {
  //     delete layer;
  //   }
  //   for (auto *act : activations)
  //   {
  //     delete act;
  //   }
  //   delete loss_function;
  // }

  void save_weights(const std::string &filename)
  {
    std::ofstream file(filename);
    if (!file.is_open())
    {
      throw std::runtime_error("No se pudo abrir el archivo para guardar los pesos.");
    }

    for (size_t l = 0; l < layers.size(); ++l)
    {
      if (layers[l]->has_weights())
      {
        const Tensor &weights = layers[l]->get_weights();
        if (weights.data.empty())
          continue; // Capa sin pesos (por ejemplo, Flatten, ReLU, etc.)

        file << "Layer " << l << ":\n";

        // Guardar la forma
        file << "Shape: ";
        for (size_t i = 0; i < weights.shape.size(); ++i)
        {
          file << weights.shape[i];
          if (i < weights.shape.size() - 1)
            file << "x";
        }
        file << "\n";

        // Guardar datos (planos, en filas de 10 por legibilidad)
        file << "Data:\n";
        for (size_t i = 0; i < weights.data.size(); ++i)
        {
          file << std::fixed << std::setprecision(6) << weights.data[i];
          if ((i + 1) % 10 == 0 || i == weights.data.size() - 1)
            file << "\n";
          else
            file << ", ";
        }

        file << "\n";
      }
    }

    file.close();
  }

  void load_weights(const std::string &filename)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      throw std::runtime_error("No se pudo abrir el archivo para cargar los pesos.");
    }

    std::string line;
    size_t current_layer = 0;
    std::vector<int> current_shape;
    std::vector<float> current_data;

    while (std::getline(file, line))
    {
      if (line.empty())
      {
        // Fin de un bloque de capa: construir tensor y asignar
        while (current_layer < layers.size() && layers[current_layer]->get_weights().data.empty())
        {
          current_layer++; // Saltar capas sin pesos
        }

        if (current_layer >= layers.size())
          break;

        Tensor tensor(current_shape, current_data);
        layers[current_layer]->set_weights(tensor);

        current_shape.clear();
        current_data.clear();
        current_layer++;
      }
      else if (line.find("Shape:") != std::string::npos)
      {
        // Parsear shape
        current_shape.clear();
        std::string shape_str = line.substr(7); // Después de "Shape: "
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, 'x'))
        {
          current_shape.push_back(std::stoi(dim));
        }
      }
      else if (line.find("Data:") != std::string::npos)
      {
        // Comienzo de los datos, no hacemos nada
        continue;
      }
      else if (line.find("Layer") != std::string::npos)
      {
        // Ignorar encabezado de capa
        continue;
      }
      else
      {
        // Parsear línea de datos numéricos
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ','))
        {
          try
          {
            current_data.push_back(std::stof(value));
          }
          catch (...)
          {
            std::cerr << "Error convirtiendo valor a float: " << value << std::endl;
          }
        }
      }
    }

    // Si hay pesos sin aplicar al final del archivo
    if (!current_data.empty() && !current_shape.empty() && current_layer < layers.size())
    {
      while (current_layer < layers.size() && layers[current_layer]->get_weights().data.empty())
      {
        current_layer++;
      }

      if (current_layer < layers.size())
      {
        Tensor tensor(current_shape, current_data);
        layers[current_layer]->set_weights(tensor);
      }
    }

    file.close();
  }
};
