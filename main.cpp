#include "models/model.hpp"
#include "utils/load_dataset.hpp"
#include "models/trainer.hpp"
#include <chrono>

std::mt19937 Layer::gen(32); // Semilla para reproducibilidad

int main()
{
  auto trainImages = loadImages2D("mnist_data/train-images.idx3-ubyte");
  auto trainLabels = loadLabels("mnist_data/train-labels.idx1-ubyte");
  auto testImages = loadImages2D("mnist_data/t10k-images.idx3-ubyte");
  auto testLabels = loadLabels("mnist_data/t10k-labels.idx1-ubyte");

  std::cout << "Cargadas " << trainImages.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << testImages.size() << " imágenes de prueba." << std::endl;

  float learning_rate = 0.001f;
  float wd = 0.0005f;

  Optimizer *adam = new Adam(learning_rate);
  CrossEntropyLoss *loss = new CrossEntropyLoss();
  ReLU *relu = new ReLU();
  Model cnn(learning_rate, adam);

  cnn.add_layer(new Conv2DLayer(1, 4, 3, 28, 28, 1, 0, relu, adam)); // out: 4 x 26 x 26
  cnn.add_layer(new PoolingLayer(4, 26, 26, 2, 2));                  // out: 4 x 13 x 13
  cnn.add_layer(new Conv2DLayer(4, 4, 3, 13, 13, 1, 0, relu, adam)); // out: 4 x 11 x 11
  cnn.add_layer(new PoolingLayer(4, 11, 11, 2, 3));                  // out: 4 x 4 x 4

  cnn.add_layer(new FlattenLayer()); // out: 4 * 4 * 4 = 64

  cnn.add_layer(new DenseLayer(64, 10, new Softmax(), adam));
  cnn.set_loss(new CrossEntropyLoss());

  auto start_time = std::chrono::high_resolution_clock::now();

  Trainer trainer(cnn, loss, adam);

  cnn.load_weights("save_models/conv2d-2-epochs.txt");
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;

  Metrics final_metrics = trainer.evaluate(testImages, testLabels);
  std::cout << "Final validation accuracy: " << final_metrics.accuracy * 100 << "%\n";

  return 0;
}