#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono> // al inicio de tu archivo, si aún no está

mt19937 Perceptron::gen(32);

int main()
{
  auto train_data = load_dataset("/home/leon/Documentos/UNSA/TOPICOS IA/MLP/mnist_data/saved_images/train");
  auto test_data = load_dataset("/home/leon/Documentos/UNSA/TOPICOS IA/MLP/mnist_data/saved_images/test");

  std::cout << "Cargadas " << train_data.first.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << test_data.first.size() << " imágenes de prueba." << std::endl;

  float learning_rate = 0.001f;
  Optimizer *sgd = new SGD(learning_rate);
  MLP mlp(learning_rate, sgd);
  mlp.add_input_layer(784, 128, new ReLU());
  mlp.add_layer(64, new ReLU());
  mlp.add_layer(10, new Softmax());
  mlp.set_loss(new CrossEntropyLoss());

  auto start_time = std::chrono::high_resolution_clock::now();

  mlp.train(20, train_data.first, train_data.second);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;

  // 4. Probar las primeras 10 muestras de test
  std::cout << "\n=== Evaluación sobre 10 muestras de test ===\n";
  for (size_t i = 0; i < 10 && i < test_data.first.size(); ++i)
  {
    const auto &x = test_data.first[i];
    int true_label = static_cast<int>(test_data.second[i]);
    int pred = mlp.predict(x);

    std::cout << "Muestra " << i
              << " — Verdadero: " << true_label
              << ", Predicción: " << pred << "\n";
  }
  mlp.evaluate(test_data.first, test_data.second);
  return 0;
}
