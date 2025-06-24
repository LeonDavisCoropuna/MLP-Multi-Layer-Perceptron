// trainer.hpp
#pragma once
#include "model.hpp"
#include <fstream>
#include <sstream>
#include "../utils/data_loader.hpp"
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

  Metrics compute_metrics(DataLoader &loader)
  {
    Metrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool is_binary = (model.layers.back()->output_size() == 1);

    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    int true_negatives = 0;
    float total_loss = 0.0f;
    int total_samples = 0;

    loader.reset(); // No barajar para evaluación

    while (loader.has_next())
    {
      auto [batch_X, batch_Y] = loader.next_batch();
      Tensor outputs = model.forward(batch_X); // outputs: [batch_size, output_dim]
      size_t batch_size = batch_X.shape[0];
      int output_dim = model.layers.back()->output_size();
      total_samples += batch_size;

      for (size_t i = 0; i < batch_size; ++i)
      {
        float y_true = batch_Y[i];
        Tensor target_vec;

        if (is_binary)
        {
          target_vec.data = {y_true};
        }
        else
        {
          target_vec.data.assign(output_dim, 0.0f);
          target_vec.data[static_cast<int>(y_true)] = 1.0f;
        }

        // Índice base del output actual en la salida
        std::vector<float> output_row(outputs.data.begin() + i * output_dim,
                                      outputs.data.begin() + (i + 1) * output_dim);

        total_loss += loss_function->compute(output_row, target_vec.data);

        // Calcular predicción
        int predicted_class;
        if (is_binary)
        {
          predicted_class = (output_row[0] > 0.5f) ? 1 : 0;
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
              output_row.begin(), std::max_element(output_row.begin(), output_row.end())));
          if (predicted_class == static_cast<int>(y_true))
          {
            true_positives++;
          }
        }
      }
    }

    metrics.loss = total_loss / total_samples;

    if (is_binary)
    {
      metrics.accuracy = static_cast<float>(true_positives + true_negatives) / total_samples;

      float precision_denominator = (true_positives + false_positives);
      metrics.precision = precision_denominator > 0 ? static_cast<float>(true_positives) / precision_denominator : 0.0f;

      float recall_denominator = (true_positives + false_negatives);
      metrics.recall = recall_denominator > 0 ? static_cast<float>(true_positives) / recall_denominator : 0.0f;

      float f1_denominator = metrics.precision + metrics.recall;
      metrics.f1 = f1_denominator > 0 ? 2 * (metrics.precision * metrics.recall) / f1_denominator : 0.0f;
    }
    else
    {
      metrics.accuracy = static_cast<float>(true_positives) / total_samples;
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

  void process_batch(const Tensor &batch_X,
                     const std::vector<float> &batch_Y,
                     bool is_binary,
                     float &batch_loss)
  {
    // Limpiar gradientes
    for (auto &layer : model.layers)
    {
      layer->zero_grad();
    }

    // Forward pass con todo el batch
    Tensor outputs = model.forward(batch_X); // outputs: [batch_size, output_dim]
    size_t batch_size = batch_X.shape[0];
    int output_dim = model.layers.back()->output_size();

    // Construir vector de targets como tensor batch
    Tensor targets;
    if (is_binary)
    {
      targets = Tensor({(int)batch_size, 1});
      for (size_t i = 0; i < batch_size; ++i)
      {
        targets.data[i] = batch_Y[i];
      }
    }
    else
    {
      targets = Tensor({(int)batch_size, output_dim});
      for (size_t i = 0; i < batch_size; ++i)
      {
        int label = static_cast<int>(batch_Y[i]);
        targets.data[i * output_dim + label] = 1.0f; // one-hot encoding
      }
    }

    // Calcular pérdida para todo el batch
    batch_loss += loss_function->compute(outputs.data, targets.data); // ajusta si necesitas promedio

    // Backward pass con batch de targets
    model.backward_pass(targets);

    // Aplicar gradientes (puedes dividir por batch_size si tu optimizador lo requiere)
    model.apply_gradients(batch_size);
  }

public:
  Trainer(Model &mlp, Loss *loss_fn, Optimizer *optim)
      : model(mlp), loss_function(loss_fn), optimizer(optim) {}

  void train(int num_epochs,
             DataLoader &train_loader, DataLoader &test_loader,
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

      // Resetear y barajar el dataloader al inicio de cada época
      train_loader.reset(); // true => shuffle

      // Modo entrenamiento
      for (auto &layer : model.layers)
      {
        layer->set_training(true);
      }

      // Iterar sobre batches del dataloader
      while (train_loader.has_next())
      {
        auto [batch_X, batch_Y] = train_loader.next_batch();

        float batch_loss = 0.0f;
        process_batch(batch_X, batch_Y, is_binary, batch_loss);
        total_loss += batch_loss;

        optimizer->increment_t();
      }

      // Modo evaluación
      for (auto &layer : model.layers)
      {
        layer->set_training(false);
      }

      // Calcular métricas (fuera de DataLoader aún)
      Metrics train_metrics = compute_metrics(train_loader);
      Metrics val_metrics = compute_metrics(test_loader);

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

  Metrics evaluate(DataLoader loader)
  {
    for (auto layer : model.layers)
    {
      layer->set_training(false);
    }
    return compute_metrics(loader);
  }
};