#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>

mt19937 SingleLayerPerceptron::gen(32);

int main()
{
  float learning_rate = 0.001f;
  Optimizer *sgd = new SGD(learning_rate);
  MLP mlp(learning_rate, sgd);
  mlp.add_input_layer(784, 128, new ReLU());
  mlp.add_layer(64, new ReLU());
  mlp.add_layer(10, new Softmax());
  mlp.set_loss(new CrossEntropyLoss());

  // mlp.load_model_weights("/home/leon/Documentos/UNSA/TOPICOS IA/MLP/save_models/minst_weights.txt");

  //flatten_image_to_vector_and_predict("numbers/Captura desde 2025-05-26 16-10-36.png", mlp);
  
  auto test_data = load_dataset("/home/leon/Documentos/UNSA/TOPICOS IA/MLP/dataset-testing/mnist45/mnist45");
  mlp.evaluate(test_data.first, test_data.second);

  return 0;
}