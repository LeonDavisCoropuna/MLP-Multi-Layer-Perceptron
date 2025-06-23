#include "models/model.hpp"
#include <iostream>
#include <iomanip>

std::mt19937 Layer::gen(32);

void print_tensor(const Tensor& t) {
    std::cout << "Shape [";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i < t.shape.size()-1 ? "x" : "");
    }
    std::cout << "]:\n";
    
    for (size_t i = 0; i < t.data.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << t.data[i] << " ";
        if ((i+1) % t.shape.back() == 0) std::cout << "\n";
    }
}

int main() {
    // // Entrada 1x6x6 (patrón diagonal)
    // Tensor input({1, 6, 6});
    // for (int i = 0; i < 6; ++i) {
    //     for (int j = 0; j < 6; ++j) {
    //         input.data[i*6 + j] = (i == j) ? 1.0f : 0.0f;
    //     }
    // }

    // MLP mlp(0, nullptr);
    
    // // Bloque extractor de características
    // mlp.add_layer(new Conv2DLayer(1, 4, 3, 6, 6, 1, 0, new ReLU())); // 4x4x4
    // mlp.add_layer(new PoolingLayer(4, 4, 4, 2, 2));            // 4x2x2
    
    // // Transformación para fully-connected
    // mlp.add_layer(new FlattenLayer());                          // 16

    // // Procesamiento
    // std::cout << "=== Entrada ===\n";
    // print_tensor(input);
    
    // Tensor output = mlp.forward(input);
    
    // std::cout << "\n=== Salida Flatten ===";
    // print_tensor(output);

    return 0;
}