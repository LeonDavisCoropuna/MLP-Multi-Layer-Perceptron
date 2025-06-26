#include "models/model.hpp"
#include "utils/load_dataset.hpp"
#include "models/trainer.hpp"
#include <chrono>
#include "utils/data_loader.hpp"
std::mt19937 Layer::gen(32); // Semilla para reproducibilidad

int main()
{
  auto trainImages = loadImages2D("/home/leon/Documentos/UNSA/TOPICOS IA/MLP-Multi-Layer-Perceptron/mnist_data/train-images.idx3-ubyte", 20000);
  auto trainLabels = loadLabels("/home/leon/Documentos/UNSA/TOPICOS IA/MLP-Multi-Layer-Perceptron/mnist_data/train-labels.idx1-ubyte", 20000);
  auto testImages = loadImages2D("/home/leon/Documentos/UNSA/TOPICOS IA/MLP-Multi-Layer-Perceptron/mnist_data/t10k-images.idx3-ubyte");
  auto testLabels = loadLabels("/home/leon/Documentos/UNSA/TOPICOS IA/MLP-Multi-Layer-Perceptron/mnist_data/t10k-labels.idx1-ubyte");

  std::cout << "Cargadas " << trainImages.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << testImages.size() << " imágenes de prueba." << std::endl;

  DataLoader train_loader(trainImages, trainLabels, 32); // batch size 32
  DataLoader test_loader(testImages, testLabels, 32);    // batch size 32

  float learning_rate = 0.001f;
  float wd = 0.0005f;

  ReLU *relu = new ReLU();
  Softmax *softmax = new Softmax();
  CrossEntropyLoss *loss = new CrossEntropyLoss();

  Model cnn(learning_rate, nullptr); // El modelo ya no guarda un Adam global

  /*
  [32,1,28,28] //input con batch 32

  salida conv2d [32,8,26,26]
  salida pooling [32,8,13,13]
  salida tokenizer [32,13*13+16,8] //16 es el numero de tokens
  salida transformer [32,13*13+16,8]
  salida del projector [32,8,13,13] //devuelve al espacio original
  salida del flatten [32,13*13*8]
  salida del dense [32,10]
  */

  cnn.add_layer(new Conv2DLayer(1, 8, 3, 28, 28, 1, 0, relu, new SGD(learning_rate)));
  cnn.add_layer(new PoolingLayer(8, 26, 26, 2, 2));
  // cnn.add_layer(new VisionTransformerBlock(8, 16, 8, false, relu, new SGD(learning_rate)));
  cnn.add_layer(new FlattenLayer());
  cnn.add_layer(new DenseLayer(8 * 13 * 13, 10, softmax, new SGD(learning_rate)));

  cnn.set_loss(loss);

  auto start_time = std::chrono::high_resolution_clock::now();

  Trainer trainer(cnn, loss, new SGD(learning_rate));

  trainer.train(10, train_loader, test_loader, 32, "training_log_sgd.csv");
  // Tensor sss = cnn.forward(train_loader.next_batch().first);
  // cnn.load_weights("save_models/conv2d-2-epochs.txt");
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;
  
  std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;

  Metrics final_metrics = trainer.evaluate(test_loader);
  std::cout << "Final validation accuracy: " << final_metrics.accuracy * 100 << "%\n";

  return 0;
}