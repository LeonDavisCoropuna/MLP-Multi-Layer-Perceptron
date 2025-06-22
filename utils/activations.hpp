#pragma once
#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <random>

class ActivationFunction
{
public:
  virtual ~ActivationFunction() = default;

  // Versión escalar (para compatibilidad)
  virtual float activate(float x) const = 0;
  virtual float derivative(float x) const = 0;

  // Versión tensor (principal)
  virtual Tensor activate(const Tensor &x) const
  {
    Tensor result(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i)
    {
      result.data[i] = activate(x.data[i]);
    }
    return result;
  }

  virtual Tensor derivative(const Tensor &x) const
  {
    Tensor result(x.shape);
    for (size_t i = 0; i < x.data.size(); ++i)
    {
      result.data[i] = derivative(x.data[i]);
    }
    return result;
  }

  // Inicialización de pesos (ahora trabaja con tensores)
  virtual void initialize_weights(Tensor &weights, int num_inputs, std::mt19937 &gen) const = 0;

  // Para activaciones especiales como Softmax
  virtual bool requires_special_output_gradient() const { return false; }
  virtual Tensor activate_vector(const Tensor &x) const
  {
    throw std::runtime_error("activate_vector no implementado para esta función de activación");
  }
};

// Implementación de ReLU con tensores
class ReLU : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return (x > 0) ? x : 0;
  }

  float derivative(float x) const override
  {
    return (x > 0) ? 1.0f : 0.0f;
  }

  void initialize_weights(Tensor &weights, int num_inputs, std::mt19937 &gen) const override
  {
    // Inicialización He (He et al., 2015) para ReLU
    float stddev = sqrt(2.0f / num_inputs);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto &w : weights.data)
    {
      w = dist(gen);
    }
  }
};

// Implementación de Tanh con tensores
class Tanh : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return tanh(x);
  }

  float derivative(float x) const override
  {
    return 1.0f - tanh(x) * tanh(x);
  }

  void initialize_weights(Tensor &weights, int num_inputs, std::mt19937 &gen) const override
  {
    // Inicialización Xavier/Glorot
    float limit = sqrt(6.0f / num_inputs);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weights.data)
    {
      w = dist(gen);
    }
  }
};

// Implementación de Sigmoid con tensores
class Sigmoid : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return 1.0f / (1.0f + exp(-x));
  }

  float derivative(float x) const override
  {
    float sig = activate(x);
    return sig * (1 - sig);
  }

  void initialize_weights(Tensor &weights, int num_inputs, std::mt19937 &gen) const override
  {
    // Inicialización Xavier/Glorot
    float limit = sqrt(6.0f / num_inputs);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weights.data)
    {
      w = dist(gen);
    }
  }
};

// Implementación de Softmax con tensores
class Softmax : public ActivationFunction
{
public:
  float activate(float x) const override
  {
    return x; // No aplica softmax escalar
  }

  float derivative(float x) const override
  {
    return 1.0f; // No usada directamente
  }

  void initialize_weights(Tensor &weights, int num_inputs, std::mt19937 &gen) const override
  {
    // Inicialización Xavier/Glorot
    float limit = sqrt(6.0f / num_inputs);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto &w : weights.data)
    {
      w = dist(gen);
    }
  }

  bool requires_special_output_gradient() const override { return true; }

  Tensor activate_vector(const Tensor &x) const override
  {
    if (x.shape.size() != 1)
    {
      throw std::runtime_error("Softmax requiere un tensor 1D");
    }

    Tensor result(x.shape);

    // Estabilización numérica: restar el máximo
    float max_val = *std::max_element(x.data.begin(), x.data.end());
    float sum_exp = 0.0f;

    // Calcular exponenciales
    for (size_t i = 0; i < x.data.size(); ++i)
    {
      result.data[i] = exp(x.data[i] - max_val);
      sum_exp += result.data[i];
    }

    // Normalizar
    for (size_t i = 0; i < x.data.size(); ++i)
    {
      result.data[i] /= sum_exp;
    }

    return result;
  }

  // Sobreescribir activate para manejar tanto vectores como escalares
  Tensor activate(const Tensor &x) const override
  {
    if (x.shape.size() == 1)
    {
      return activate_vector(x);
    }
    return ActivationFunction::activate(x); // Usar implementación base para escalares
  }
};