#include "models/model.hpp"
#include "utils/load_dataset.hpp"
#include "models/trainer.hpp"
#include <chrono>
#include "utils/data_loader.hpp"
std::mt19937 Layer::gen(32); // Semilla para reproducibilidad

int main()
{
  Conv2DLayer conv(1, 5, 3, 6, 6, 1, 0, new Tanh());

  // MaxPooling: canales=5 (output de conv), input x3 (porque kernel=3 y stride=2), pool=2x2, stride=2
  PoolingLayer maxpool(5, 4, 4, 2, 2, PoolingType::AVG); // asume que conv deja 3x3

  // Flatten Layer
  FlattenLayer flatten;

  //[batch_size, channels, h, w]
  Tensor x({1, 1, 6, 6},
           {1, 2, 3, 4, 5, 2,
            6, 7, 8, 9, 1, 3,
            2, 3, 4, 5, 6, 4,
            7, 8, 9, 1, 2, 1,
            3, 4, 5, 6, 7, 2,
            7, 5, 6, 1, 2, 3});

  // Forward pass
  Tensor out1 = conv.forward({x});       // [4, 3, 3]
  Tensor out2 = maxpool.forward({out1}); // [4, 1, 1]
  Tensor out3 = flatten.forward({out2}); // [4] o [1, 4] si mantienes batch

  // Imprimir el resultado final
  std::cout << "Output shape: " << std::endl;
  out3.print_shape();
  std::cout << "Output flatten: ";
  for (float val : out3.data)
  {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  return 0;
}