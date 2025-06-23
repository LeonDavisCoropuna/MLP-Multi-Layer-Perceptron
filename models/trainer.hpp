// trainer.hpp
#pragma once
#include "model.hpp"
#include <fstream>
#include <sstream>

struct Metrics
{
  float loss;
  float accuracy;
  float precision;
  float recall;
  float f1;
};

class Trainer
{
private:
  Model &model;
  Loss *loss_function;
  Optimizer *optimizer;

  Metrics compute_metrics(const std::vector<Tensor> &X, const std::vector<float> &Y)
  {
    Metrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool is_binary = (model.layers.back()->output_size() == 1);

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
        target_vec.data.assign(model.layers.back()->output_size(), 0.0f);
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

  void print_metrics(int epoch, const Metrics &train, const Metrics &val)
  {
    std::cout << "Epoch " << epoch + 1 << "\n"
              << "  Train Loss: " << train.loss << "\n"
              << "  Train Accuracy: " << train.accuracy * 100 << "%\n"
              << "  Train Precision: " << train.precision << "\n"
              << "  Train Recall: " << train.recall << "\n"
              << "  Train F1: " << train.f1 << "\n"
              << "  Val Loss: " << val.loss << "\n"
              << "  Val Accuracy: " << val.accuracy * 100 << "%\n"
              << "  Val Precision: " << val.precision << "\n"
              << "  Val Recall: " << val.recall << "\n"
              << "  Val F1: " << val.f1 << "\n"
              << std::endl;
  }

  void log_metrics(std::ofstream &log_file, int epoch, const Metrics &train, const Metrics &val)
  {
    log_file << "Epoch " << epoch + 1 << ","
             << train.loss << ","
             << train.accuracy << ","
             << train.precision << ","
             << train.recall << ","
             << train.f1 << ","
             << val.loss << ","
             << val.accuracy << ","
             << val.precision << ","
             << val.recall << ","
             << val.f1 << "\n";
  }

  void process_batch(const std::vector<Tensor> &batch_X,
                     const std::vector<float> &batch_Y,
                     bool is_binary,
                     float &batch_loss)
  {

    // Limpiar gradientes
    for (auto &layer : model.layers)
    {
      layer->zero_grad();
    }

    // Procesar cada muestra del batch
    for (size_t i = 0; i < batch_X.size(); i++)
    {
      Tensor outputs = model.forward(batch_X[i]);
      float y_true = batch_Y[i];

      // Crear target vector
      Tensor target_vec;
      if (is_binary)
      {
        target_vec.data = {y_true};
      }
      else
      {
        target_vec.data.assign(model.layers.back()->output_size(), 0.0f);
        target_vec.data[static_cast<int>(y_true)] = 1.0f;
      }

      // Calcular pérdida
      batch_loss += loss_function->compute(outputs.data, target_vec.data);

      // Backward pass abstracto
      model.backward_pass(target_vec);

      // Acumular gradientes
      for (auto &layer : model.layers)
      {
        layer->accumulate_gradients();
      }
    }

    // Aplicar gradientes
    model.apply_gradients(batch_X.size());
  }

public:
  Trainer(Model &mlp, Loss *loss_fn, Optimizer *optim)
      : model(mlp), loss_function(loss_fn), optimizer(optim) {}

  void train(int num_epochs,
             const std::vector<Tensor> &X_train, const std::vector<float> &Y_train,
             const std::vector<Tensor> &X_val, const std::vector<float> &Y_val,
             int batch_size = 1,
             const std::string &log_filepath = "",
             bool verbose = true)
  {

    bool is_binary = (model.layers.back()->output_size() == 1);
    std::ofstream log_file;

    // Configurar archivo de log
    if (!log_filepath.empty())
    {
      log_file.open(log_filepath);
      if (log_file.is_open())
      {
        log_file << "epoch,train_loss,train_acc,train_prec,train_rec,train_f1,"
                 << "val_loss,val_acc,val_prec,val_rec,val_f1\n";
      }
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
      float total_loss = 0.0f;

      // Shuffle
      std::vector<size_t> indices(X_train.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), model.layers[0]->gen);

      // Modo entrenamiento
      for (auto layer : model.layers)
      {
        layer->set_training(true);
      }

      // Procesar en batches
      for (size_t batch_start = 0; batch_start < X_train.size(); batch_start += batch_size)
      {
        size_t batch_end = std::min(batch_start + batch_size, X_train.size());
        std::vector<Tensor> batch_X(batch_end - batch_start);
        std::vector<float> batch_Y(batch_end - batch_start);

        // Preparar batch
        for (size_t i = batch_start; i < batch_end; i++)
        {
          size_t idx = indices[i];
          batch_X[i - batch_start] = X_train[idx];
          batch_Y[i - batch_start] = Y_train[idx];
        }

        float batch_loss = 0.0f;
        process_batch(batch_X, batch_Y, is_binary, batch_loss);
        total_loss += batch_loss;
        optimizer->increment_t();
      }

      // Modo evaluación
      for (auto layer : model.layers)
      {
        layer->set_training(false);
      }

      // Calcular métricas
      Metrics train_metrics = compute_metrics(X_train, Y_train);
      Metrics val_metrics = compute_metrics(X_val, Y_val);

      // Logging
      if (verbose)
      {
        print_metrics(epoch, train_metrics, val_metrics);
      }

      if (log_file.is_open())
      {
        log_metrics(log_file, epoch, train_metrics, val_metrics);
      }
    }

    if (log_file.is_open())
    {
      log_file.close();
    }
  }

  Metrics evaluate(const std::vector<Tensor> &X_test, const std::vector<float> &Y_test)
  {
    for (auto layer : model.layers)
    {
      layer->set_training(false);
    }
    return compute_metrics(X_test, Y_test);
  }
};