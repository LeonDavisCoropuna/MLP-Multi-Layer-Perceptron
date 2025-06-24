#pragma once
#include "tensor.hpp"
#include <cmath>
#include <omp.h>

class Optimizer
{
public:
  virtual ~Optimizer() = default;
  virtual void update(Tensor &weights, const Tensor &gradients_weights,
                      Tensor &biases, const Tensor &gradients_biases) = 0;
  virtual void increment_t() {};
};

// SGD (Stochastic Gradient Descent)
class SGD : public Optimizer
{
private:
  float learning_rate;
  float weight_decay;

public:
  explicit SGD(float lr, float wd = 0.0f) : learning_rate(lr), weight_decay(wd) {}

  void update(Tensor &weights, const Tensor &gradients_weights,
              Tensor &biases, const Tensor &gradients_biases) override
  {

    if (weights.shape != gradients_weights.shape ||
        biases.shape != gradients_biases.shape)
    {
      throw std::runtime_error("Shape mismatch in SGD optimizer");
    }

#pragma omp parallel for
    for (size_t i = 0; i < weights.data.size(); ++i)
    {
      // Aplicar weight decay: grad += weight_decay * weight
      float grad_with_decay = gradients_weights.data[i] + weight_decay * weights.data[i];
      weights.data[i] -= learning_rate * grad_with_decay;
    }

#pragma omp parallel for
    for (size_t i = 0; i < biases.data.size(); ++i)
    {
      biases.data[i] -= learning_rate * gradients_biases.data[i]; // No weight decay para biases
    }
  }
};

// Adam Optimizer
class Adam : public Optimizer
{
private:
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
  Tensor m_weights;
  Tensor v_weights;
  Tensor m_biases;
  Tensor v_biases;
  int t;

public:
  explicit Adam(float lr = 0.001f, float wd = 0.0f, float b1 = 0.9f, float b2 = 0.999f,
                float eps = 1e-8f)
      : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd), t(1) {}

  void update(Tensor &weights, const Tensor &gradients_weights,
              Tensor &biases, const Tensor &gradients_biases)
  {
    if (weights.shape != gradients_weights.shape || biases.shape != gradients_biases.shape)
    {
      std::cerr << "Error: shape mismatch in Adam::update()\n";
      std::cerr << "weights.shape: ";
      for (int s : weights.shape)
        std::cerr << s << " ";
      std::cerr << "\ngradients_weights.shape: ";
      for (int s : gradients_weights.shape)
        std::cerr << s << " ";
      std::cerr << std::endl;
      std::exit(1); // forzar salida
    }

    // Inicialización en la primera iteración
    if (m_weights.shape.empty())
    {
      m_weights = Tensor(weights.shape);
      v_weights = Tensor(weights.shape);
      m_biases = Tensor(biases.shape);
      v_biases = Tensor(biases.shape);
    }

#pragma omp parallel for
    for (size_t i = 0; i < weights.data.size(); ++i)
    {
      // Aplicar weight decay al gradiente
      float grad_with_decay = gradients_weights.data[i] + weight_decay * weights.data[i];

      m_weights.data[i] = beta1 * m_weights.data[i] + (1.0f - beta1) * grad_with_decay;
      v_weights.data[i] = beta2 * v_weights.data[i] + (1.0f - beta2) * grad_with_decay * grad_with_decay;

      float m_hat = m_weights.data[i] / (1.0f - std::pow(beta1, t));
      float v_hat = v_weights.data[i] / (1.0f - std::pow(beta2, t));

      weights.data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

#pragma omp parallel for
    for (size_t i = 0; i < biases.data.size(); ++i)
    {
      // Biases (sin weight decay)
      m_biases.data[i] = beta1 * m_biases.data[i] + (1.0f - beta1) * gradients_biases.data[i];
      v_biases.data[i] = beta2 * v_biases.data[i] + (1.0f - beta2) * gradients_biases.data[i] * gradients_biases.data[i];

      float m_hat_bias = m_biases.data[i] / (1.0f - std::pow(beta1, t));
      float v_hat_bias = v_biases.data[i] / (1.0f - std::pow(beta2, t));
      biases.data[i] -= learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);
    }
  }

  void increment_t() override { ++t; }
};

// RMSProp Optimizer
class RMSprop : public Optimizer
{
private:
  float learning_rate;
  float beta;
  float epsilon;
  float weight_decay;
  Tensor cache_weights;
  Tensor cache_biases;

public:
  explicit RMSprop(float lr = 0.001f, float b = 0.9f, float eps = 1e-8f, float wd = 0.0f)
      : learning_rate(lr), beta(b), epsilon(eps), weight_decay(wd) {}

  void update(Tensor &weights, const Tensor &gradients_weights,
              Tensor &biases, const Tensor &gradients_biases) override
  {

    // Inicialización en la primera iteración
    if (cache_weights.shape.empty())
    {
      cache_weights = Tensor(weights.shape);
      cache_biases = Tensor(biases.shape);
    }

#pragma omp parallel for
    for (size_t i = 0; i < weights.data.size(); ++i)
    {
      // Aplicar weight decay al gradiente
      float grad_with_decay = gradients_weights.data[i] + weight_decay * weights.data[i];

      cache_weights.data[i] = beta * cache_weights.data[i] +
                              (1.0f - beta) * grad_with_decay * grad_with_decay;
      weights.data[i] -= learning_rate * grad_with_decay /
                         (std::sqrt(cache_weights.data[i]) + epsilon);
    }

#pragma omp parallel for
    for (size_t i = 0; i < biases.data.size(); ++i)
    {
      // Biases (sin weight decay)
      cache_biases.data[i] = beta * cache_biases.data[i] +
                             (1.0f - beta) * gradients_biases.data[i] * gradients_biases.data[i];
      biases.data[i] -= learning_rate * gradients_biases.data[i] /
                        (std::sqrt(cache_biases.data[i]) + epsilon);
    }
  }
};