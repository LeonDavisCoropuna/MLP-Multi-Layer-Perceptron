#include <vector>
#include <cmath> // Para log(), pow(), etc.

#include <vector>

class Optimizer
{
public:
  virtual ~Optimizer() = default;
  virtual void update(std::vector<float> &weights, std::vector<float> &gradients_weights,
                      float &bias, float gradient_bias) = 0;
};

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
};

class Adam : public Optimizer
{
private:
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  std::vector<float> m_weights; // Primer momento (media)
  std::vector<float> v_weights; // Segundo momento (varianza)
  float m_bias;
  float v_bias;
  int t; // Paso de tiempo

public:
  explicit Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
      : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

  void update(std::vector<float> &weights, std::vector<float> &gradients_weights,
              float &bias, float gradient_bias) override
  {
    // Inicializar momentos en la primera iteración
    if (m_weights.empty())
    {
      m_weights.resize(weights.size(), 0.0f);
      v_weights.resize(weights.size(), 0.0f);
      m_bias = 0.0f;
      v_bias = 0.0f;
    }

    t++;

    // Actualizar pesos
    for (size_t i = 0; i < weights.size(); ++i)
    {
      m_weights[i] = beta1 * m_weights[i] + (1 - beta1) * gradients_weights[i];
      v_weights[i] = beta2 * v_weights[i] + (1 - beta2) * gradients_weights[i] * gradients_weights[i];

      // Corrección de bias
      float m_hat = m_weights[i] / (1 - std::pow(beta1, t));
      float v_hat = v_weights[i] / (1 - std::pow(beta2, t));

      weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    // Actualizar bias
    m_bias = beta1 * m_bias + (1 - beta1) * gradient_bias;
    v_bias = beta2 * v_bias + (1 - beta2) * gradient_bias * gradient_bias;

    float m_hat_bias = m_bias / (1 - std::pow(beta1, t));
    float v_hat_bias = v_bias / (1 - std::pow(beta2, t));

    bias -= learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);
  }
};