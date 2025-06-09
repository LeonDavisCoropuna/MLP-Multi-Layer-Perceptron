#include <vector>
#include <cmath> // Para sqrt(), pow()

class Optimizer
{
public:
  virtual ~Optimizer() = default;
  virtual void update(std::vector<float> &weights, std::vector<float> &gradients_weights,
                      float &bias, float gradient_bias) = 0;
  virtual void increment_t() {};
};

// SGD
class SGD : public Optimizer
{
private:
  float learning_rate;

public:
  explicit SGD(float lr) : learning_rate(lr) {}

  void update(std::vector<float> &weights, std::vector<float> &gradients_weights,
              float &bias, float gradient_bias) override
  {
    for (size_t i = 0; i < weights.size(); ++i)
    {
      weights[i] -= learning_rate * gradients_weights[i];
    }
    bias -= learning_rate * gradient_bias;
  }
  void increment_t() {};
};

// Adam
class Adam : public Optimizer
{
private:
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  std::vector<float> m_weights; // Primer momento (gradientes)
  std::vector<float> v_weights; // Segundo momento (gradientes al cuadrado)
  float m_bias;
  float v_bias;
  int t;

public:
  explicit Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
      : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps),
        m_bias(0.0f), v_bias(0.0f), t(1) {}

  void update(std::vector<float> &weights, std::vector<float> &gradients_weights,
              float &bias, float gradient_bias) override
  {
    // Inicializar m y v si es la primera vez
    if (m_weights.empty())
    {
      m_weights.resize(weights.size(), 0.0f);
      v_weights.resize(weights.size(), 0.0f);
    }

    // Actualizar pesos
    for (size_t i = 0; i < weights.size(); ++i)
    {
      m_weights[i] = beta1 * m_weights[i] + (1.0f - beta1) * gradients_weights[i];
      v_weights[i] = beta2 * v_weights[i] + (1.0f - beta2) * gradients_weights[i] * gradients_weights[i];

      // Corrección de bias
      float m_hat = m_weights[i] / (1.0f - std::pow(beta1, t));
      float v_hat = v_weights[i] / (1.0f - std::pow(beta2, t));

      // Actualizar peso
      weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    // Actualizar bias (similar a los pesos)
    m_bias = beta1 * m_bias + (1.0f - beta1) * gradient_bias;
    v_bias = beta2 * v_bias + (1.0f - beta2) * gradient_bias * gradient_bias;

    float m_hat_bias = m_bias / (1.0f - std::pow(beta1, t));
    float v_hat_bias = v_bias / (1.0f - std::pow(beta2, t));

    bias -= learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);
  }
  void increment_t() { ++t; }
};

// RMSProp
class RMSprop : public Optimizer
{
private:
  float learning_rate;
  float beta;    // tasa de decaimiento
  float epsilon; // para estabilidad numérica
  std::vector<float> cache_weights;
  float cache_bias;

public:
  explicit RMSprop(float lr = 0.001f, float b = 0.9f, float eps = 1e-8f)
      : learning_rate(lr), beta(b), epsilon(eps), cache_bias(0.0f) {}

  void update(std::vector<float> &weights, std::vector<float> &gradients_weights,
              float &bias, float gradient_bias) override
  {
    if (cache_weights.empty())
    {
      cache_weights.resize(weights.size(), 0.0f);
    }

    for (size_t i = 0; i < weights.size(); ++i)
    {
      cache_weights[i] = beta * cache_weights[i] + (1.0f - beta) * gradients_weights[i] * gradients_weights[i];
      weights[i] -= learning_rate * gradients_weights[i] / (std::sqrt(cache_weights[i]) + epsilon);
    }

    cache_bias = beta * cache_bias + (1.0f - beta) * gradient_bias * gradient_bias;
    bias -= learning_rate * gradient_bias / (std::sqrt(cache_bias) + epsilon);
  }
  void increment_t() {};
};
