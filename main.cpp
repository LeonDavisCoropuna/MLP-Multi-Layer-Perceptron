#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>

std::mt19937 Layer::gen(32); // Semilla para reproducibilidad

int main()
{
  auto trainImages = loadImages2D("mnist_data/train-images.idx3-ubyte", 2000);
  auto trainLabels = loadLabels("mnist_data/train-labels.idx1-ubyte", 2000);
  auto testImages = loadImages2D("mnist_data/t10k-images.idx3-ubyte", 500);
  auto testLabels = loadLabels("mnist_data/t10k-labels.idx1-ubyte", 500);

  std::cout << "Cargadas " << trainImages.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << testImages.size() << " imágenes de prueba." << std::endl;

  float learning_rate = 0.001f;
  float wd = 0.0005f;

  Optimizer *adam = new SGD(learning_rate);
  MLP mlp(learning_rate, adam);

  // Input: [1, 28, 28]

  // Input: 1×28×28
  mlp.add_layer(new Conv2DLayer(1, 32, 3, 28, 28, new ReLU(), adam));  // 32×26×26
  mlp.add_layer(new PoolingLayer(32, 26, 26, 2, 2));                   // 32×13×13
  mlp.add_layer(new Conv2DLayer(32, 32, 3, 13, 13, new ReLU(), adam)); // 32×11×11
  mlp.add_layer(new PoolingLayer(32, 11, 11, 2, 1));                   // 32×10×10
  mlp.add_layer(new Conv2DLayer(32, 32, 3, 10, 10, new ReLU(), adam)); // 32×8×8
  mlp.add_layer(new PoolingLayer(32, 8, 8, 2, 2));                     // 32×4×4
  mlp.add_layer(new Conv2DLayer(32, 32, 3, 4, 4, new ReLU(), adam));   // 32×2×2
  mlp.add_layer(new FlattenLayer());
  mlp.add_layer(new DenseLayer(32 * 2 * 2, 10, new Softmax(), adam));
  
  // mlp.add_layer(new DropoutLayer(0.2));
  // mlp.add_layer(new DenseLayer(64, 32, new ReLU(), adam));
  // mlp.add_layer(new DropoutLayer(0.2));
  // mlp.add_layer(new DenseLayer(32, 10, new Softmax(), adam));
  mlp.set_loss(new CrossEntropyLoss());

  auto start_time = std::chrono::high_resolution_clock::now();

  mlp.train(10, trainImages, trainLabels, testImages, testLabels, 32, "output/asdasd.txt");
  // mlp.save_weights("save_models/drop02-wd-0005-arch784x64x32x10.txt");
  // mlp.load_weights("save_models/drop02-arch784x64x32x10.txt");
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;

  mlp.evaluate(testImages, testLabels);

  return 0;
}