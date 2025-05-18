#include "models/MLP.hpp"
#include <chrono>

mt19937 Perceptron::gen(32);

int main()
{
  float learning_rate = 0.05f; // Tasa de aprendizaje más alta para XOR
  vector<vector<float>> X = {
      {0, 0},
      {0, 1},
      {1, 0},
      {1, 1}};
  vector<float> Y_and = {0, 0, 0, 1};
  vector<float> Y_or = {0, 1, 1, 1};
  vector<float> Y_xor = {0, 1, 1, 0};

  vector<float> Y = Y_xor;

  Optimizer *sgd = new SGD(learning_rate);
  MLP mlp(learning_rate, sgd);
  mlp.add_input_layer(2, 4, new ReLU());
  mlp.add_layer(1, new ReLU());
  mlp.set_loss(new MSELoss());

  auto start_time = std::chrono::high_resolution_clock::now();
  mlp.train(150, X, Y);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_time - start_time;

  std::cout << "Tiempo de entrenamiento: " << duration.count() << " segundos\n";

  std::cout << "\n=== Resultados XOR ===\n";
  for (size_t i = 0; i < X.size(); ++i)
  {
    float pred = mlp.predict(X[i]); // Usar [0] porque la salida es un vector de 1 elemento
    std::cout << "Entrada: [" << X[i][0] << ", " << X[i][1] << "] "
              << "-> Predicción: " << (pred > 0.5f ? 1 : 0)
              << " (Valor real: " << Y[i] << ")\n";
  }

  return 0;
}