#include "perceptron.hpp"
#include "../utils/optimizer.hpp"
class SingleLayerPerceptron
{
public:
  vector<Perceptron *> list_perceptrons;
  ActivationFunction *activation;
  float learning_rate;
  vector<float> outputs_layer;
  vector<float> inputs_layer;
  Optimizer *optimizer;

public:
  SingleLayerPerceptron(int num_neurons, int num_inputs,
                        ActivationFunction *_activation,
                        float _learning_rate, Optimizer *_optimizer)
  {
    optimizer = _optimizer;
    list_perceptrons.resize(num_neurons);
    learning_rate = _learning_rate;
    for (int i = 0; i < num_neurons; i++)
    {
      Perceptron *p = new Perceptron(num_inputs, learning_rate);
      list_perceptrons[i] = p;
    }
    activation = _activation;
  }

  vector<float> forward(vector<float> batch_inputs)
  {
    inputs_layer = batch_inputs;
    outputs_layer.clear();
    vector<float> pre_activations;

    for (auto &perceptron : list_perceptrons)
    {
      float z = perceptron->forward(batch_inputs);
      perceptron->output = z;
      pre_activations.push_back(z);
    }

    if (dynamic_cast<Softmax *>(activation))
    {
      outputs_layer = activation->activate_vector(pre_activations);
    }
    else
    {
      for (float z : pre_activations)
      {
        outputs_layer.push_back(activation->activate(z));
      }
    }

    for (size_t i = 0; i < list_perceptrons.size(); ++i)
    {
      list_perceptrons[i]->output = outputs_layer[i];
    }

    return outputs_layer;
  }

  // Capa de salida
  void backward_output_layer(const vector<float> &targets)
  {
    for (int i = 0; i < list_perceptrons.size(); i++)
    {
      float output = list_perceptrons[i]->output;
      float error = output - targets[i];

      float delta;
      if (dynamic_cast<Softmax *>(activation))
      {
        // Caso Cross-Entropy + Softmax: gradiente = (output - target)
        delta = error; // ¡Sin multiplicar por derivative()!
      }
      else
      {
        // Caso MSE + Sigmoid/Lineal: gradiente = (output - target) * derivative(output)
        delta = error * activation->derivative(output);
      }

      list_perceptrons[i]->set_delta(delta);
    }
  }
  // Capa oculta
  void backward_hidden_layer(SingleLayerPerceptron *next_layer)
  {
    const int current_size = list_perceptrons.size();
    const int next_size = next_layer->list_perceptrons.size();

    std::vector<float> next_deltas(next_size);
    for (int j = 0; j < next_size; ++j)
    {
      next_deltas[j] = next_layer->list_perceptrons[j]->get_delta();
    }

#pragma omp parallel for
    for (int i = 0; i < current_size; ++i)
    {
      float sum = 0.0f;
      for (int j = 0; j < next_size; ++j)
      {
        sum += next_layer->list_perceptrons[j]->weights[i] * next_deltas[j];
      }

      float output = list_perceptrons[i]->output;
      list_perceptrons[i]->set_delta(sum * activation->derivative(output));
    }
  }

  void update_weights()
  {
    const float clip_value = 1.0f;

#pragma omp parallel for
    for (auto &neuron : list_perceptrons)
    {
      float gradient = neuron->get_delta();
      std::vector<float> gradients_weights;
      for (size_t i = 0; i < neuron->weights.size(); ++i)
      {
        gradients_weights.push_back(gradient * inputs_layer[i]);
      }
      float gradient_bias = gradient;

      optimizer->update(neuron->weights, gradients_weights, neuron->bias, gradient_bias);
    }
  }
  void zero_grad()
  {
    for (auto &perceptron : list_perceptrons)
    {
      perceptron->set_delta(0.0f);
    }
  }

  void serialize(std::ofstream &file) const
  {
    size_t num_perceptrons = list_perceptrons.size();
    file.write(reinterpret_cast<const char *>(&num_perceptrons), sizeof(num_perceptrons));

    for (const auto &perceptron : list_perceptrons)
    {
      perceptron->serialize(file);
    }
  }

  void deserialize(std::ifstream &file)
  {
    size_t num_perceptrons;
    file.read(reinterpret_cast<char *>(&num_perceptrons), sizeof(num_perceptrons));

    list_perceptrons.resize(num_perceptrons);
    for (auto &perceptron : list_perceptrons)
    {
      perceptron = new Perceptron(0, learning_rate);
      perceptron->deserialize(file);
    }
  }
};
