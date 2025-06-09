#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>

mt19937 Perceptron::gen(32);

int main()
{
  auto train_data = load_dataset("mnist_data/saved_images/train");
  auto test_data = load_dataset("mnist_data/saved_images/test");

  std::cout << "Cargadas " << train_data.first.size() << " imágenes de entrenamiento." << std::endl;
  std::cout << "Cargadas " << test_data.first.size() << " imágenes de prueba." << std::endl;

  float learning_rate = 0.001f;
  Optimizer *sgd = new Adam(learning_rate);
  MLP mlp(learning_rate, sgd);
  mlp.add_input_layer(784, 32, new ReLU());
  mlp.add_layer(10, new Softmax());
  mlp.set_loss(new CrossEntropyLoss());

  auto start_time = std::chrono::high_resolution_clock::now();

  mlp.train(100, train_data.first, train_data.second, 32, "output/adam-784-10-10.txt");
  //mlp.load_model_weights("save_models/minst_weights.txt");

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Tiempo total de entrenamiento: " << duration.count() << " segundos" << std::endl;

  mlp.evaluate(test_data.first, test_data.second);
  //mlp.save_model_weights("save_models/minst_weights.txt");

  return 0;
}