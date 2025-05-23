#include "models/MLP.hpp"
#include "utils/load_dataset.hpp"
#include <chrono>

mt19937 Perceptron::gen(32);

int main()
{
  float learning_rate = 0.001f;
  Optimizer *sgd = new SGD(learning_rate);
  MLP mlp(learning_rate, sgd);
  mlp.add_input_layer(784, 128, new ReLU());
  mlp.add_layer(64, new ReLU());
  mlp.add_layer(10, new Softmax());
  mlp.set_loss(new CrossEntropyLoss());

  mlp.load_model_weights("save_models/minst_weights.txt");

  flatten_image_to_vector_and_predict("numbers/test_00001_label_7.png", mlp);
  
  return 0;
}