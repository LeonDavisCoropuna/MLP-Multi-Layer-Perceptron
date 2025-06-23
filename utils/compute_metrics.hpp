#include <vector>
#include "tensor.hpp"
#include "loss.hpp"
#include <algorithm>
#include "../models/MLP.hpp"

struct Metrics
{
  float accuracy;
  float precision;
  float recall;
  float f1;
  float loss;
};

Metrics compute_metrics(const std::vector<Tensor> &X, const std::vector<float> &Y,
                        MLP &model, Loss *loss_function, bool is_binary)
{
  Metrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;
  int true_negatives = 0;
  float total_loss = 0.0f;

  for (size_t i = 0; i < X.size(); ++i)
  {
    Tensor outputs = model.forward(X[i]);
    float y_true = Y[i];

    // Calcular pérdida
    Tensor target_vec;
    if (is_binary)
    {
      target_vec.data = {y_true};
    }
    else
    {
      target_vec.data.assign(model.back()->output_size(), 0.0f);
      target_vec.data[static_cast<int>(y_true)] = 1.0f;
    }
    total_loss += loss_function->compute(outputs.data, target_vec.data);

    // Calcular predicción
    int predicted_class;
    if (is_binary)
    {
      predicted_class = (outputs.data[0] > 0.5f) ? 1 : 0;
      int true_class = static_cast<int>(y_true);

      if (predicted_class == 1 && true_class == 1)
        true_positives++;
      else if (predicted_class == 1 && true_class == 0)
        false_positives++;
      else if (predicted_class == 0 && true_class == 1)
        false_negatives++;
      else if (predicted_class == 0 && true_class == 0)
        true_negatives++;
    }
    else
    {
      predicted_class = static_cast<int>(std::distance(
          outputs.data.begin(), std::max_element(outputs.data.begin(), outputs.data.end())));
      if (predicted_class == static_cast<int>(y_true))
      {
        true_positives++;
      }
      // Para multiclase, necesitaríamos una matriz de confusión completa
      // Esta implementación simplificada solo calcula accuracy para multiclase
    }
  }

  metrics.loss = total_loss / X.size();

  if (is_binary)
  {
    // Cálculo de métricas para clasificación binaria
    metrics.accuracy = static_cast<float>(true_positives + true_negatives) / X.size();

    float precision_denominator = (true_positives + false_positives);
    metrics.precision = precision_denominator > 0 ? static_cast<float>(true_positives) / precision_denominator : 0.0f;

    float recall_denominator = (true_positives + false_negatives);
    metrics.recall = recall_denominator > 0 ? static_cast<float>(true_positives) / recall_denominator : 0.0f;

    float f1_denominator = metrics.precision + metrics.recall;
    metrics.f1 = f1_denominator > 0 ? 2 * (metrics.precision * metrics.recall) / f1_denominator : 0.0f;
  }
  else
  {
    // Para multiclase, solo calculamos accuracy (implementación simplificada)
    metrics.accuracy = static_cast<float>(true_positives) / X.size();
    metrics.precision = metrics.accuracy;
    metrics.recall = metrics.accuracy;
    metrics.f1 = metrics.accuracy;
  }

  return metrics;
}