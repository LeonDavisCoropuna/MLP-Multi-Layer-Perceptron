#pragma once
#include <vector>
#include <cmath>
#include <omp.h>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<std::vector<float>>& weights,
                       const std::vector<std::vector<float>>& gradients_weights,
                       std::vector<float>& biases,
                       const std::vector<float>& gradients_biases) = 0;
    virtual void increment_t() {};
};

// SGD (Stochastic Gradient Descent)
class SGD : public Optimizer {
private:
    float learning_rate;

public:
    explicit SGD(float lr) : learning_rate(lr) {}

    void update(std::vector<std::vector<float>>& weights,
               const std::vector<std::vector<float>>& gradients_weights,
               std::vector<float>& biases,
               const std::vector<float>& gradients_biases) override {
        
        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            // Actualizar pesos
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learning_rate * gradients_weights[i][j];
            }
            // Actualizar bias
            biases[i] -= learning_rate * gradients_biases[i];
        }
    }
};

// Adam Optimizer
class Adam : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    std::vector<std::vector<float>> m_weights;  // Primer momento
    std::vector<std::vector<float>> v_weights;  // Segundo momento
    std::vector<float> m_biases;
    std::vector<float> v_biases;
    int t;

public:
    explicit Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(1) {}

    void update(std::vector<std::vector<float>>& weights,
               const std::vector<std::vector<float>>& gradients_weights,
               std::vector<float>& biases,
               const std::vector<float>& gradients_biases) override {
        
        // Inicialización en la primera iteración
        if (m_weights.empty()) {
            m_weights.resize(weights.size(), std::vector<float>(weights[0].size(), 0.0f));
            v_weights.resize(weights.size(), std::vector<float>(weights[0].size(), 0.0f));
            m_biases.resize(biases.size(), 0.0f);
            v_biases.resize(biases.size(), 0.0f);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            // Actualizar momentos de los pesos
            for (size_t j = 0; j < weights[i].size(); ++j) {
                m_weights[i][j] = beta1 * m_weights[i][j] + (1.0f - beta1) * gradients_weights[i][j];
                v_weights[i][j] = beta2 * v_weights[i][j] + (1.0f - beta2) * gradients_weights[i][j] * gradients_weights[i][j];
                
                // Corrección de bias
                float m_hat = m_weights[i][j] / (1.0f - std::pow(beta1, t));
                float v_hat = v_weights[i][j] / (1.0f - std::pow(beta2, t));
                
                // Actualizar peso
                weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }

            // Actualizar momentos del bias
            m_biases[i] = beta1 * m_biases[i] + (1.0f - beta1) * gradients_biases[i];
            v_biases[i] = beta2 * v_biases[i] + (1.0f - beta2) * gradients_biases[i] * gradients_biases[i];
            
            // Corrección de bias y actualización
            float m_hat_bias = m_biases[i] / (1.0f - std::pow(beta1, t));
            float v_hat_bias = v_biases[i] / (1.0f - std::pow(beta2, t));
            biases[i] -= learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);
        }
        
        #pragma omp single
        {
            t++;
        }
    }

    void increment_t() override { ++t; }
};

// RMSProp Optimizer
class RMSprop : public Optimizer {
private:
    float learning_rate;
    float beta;
    float epsilon;
    std::vector<std::vector<float>> cache_weights;
    std::vector<float> cache_biases;

public:
    explicit RMSprop(float lr = 0.001f, float b = 0.9f, float eps = 1e-8f)
        : learning_rate(lr), beta(b), epsilon(eps) {}

    void update(std::vector<std::vector<float>>& weights,
               const std::vector<std::vector<float>>& gradients_weights,
               std::vector<float>& biases,
               const std::vector<float>& gradients_biases) override {
        
        // Inicialización en la primera iteración
        if (cache_weights.empty()) {
            cache_weights.resize(weights.size(), std::vector<float>(weights[0].size(), 0.0f));
            cache_biases.resize(biases.size(), 0.0f);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < weights.size(); ++i) {
            // Actualizar cache y pesos
            for (size_t j = 0; j < weights[i].size(); ++j) {
                cache_weights[i][j] = beta * cache_weights[i][j] + 
                                     (1.0f - beta) * gradients_weights[i][j] * gradients_weights[i][j];
                weights[i][j] -= learning_rate * gradients_weights[i][j] / 
                                (std::sqrt(cache_weights[i][j]) + epsilon);
            }

            // Actualizar cache y bias
            cache_biases[i] = beta * cache_biases[i] + 
                             (1.0f - beta) * gradients_biases[i] * gradients_biases[i];
            biases[i] -= learning_rate * gradients_biases[i] / 
                         (std::sqrt(cache_biases[i]) + epsilon);
        }
    }
};