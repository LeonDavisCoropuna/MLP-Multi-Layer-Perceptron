#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include "tensor.hpp" // Asegúrate de tener una clase Tensor con soporte para dimensiones

class DataLoader
{
private:
  const std::vector<Tensor> &images;
  const std::vector<float> &labels;
  size_t batch_size;
  size_t index;
  std::vector<size_t> indices;
  std::mt19937 rng;

public:
  DataLoader(const std::vector<Tensor> &imgs, const std::vector<float> &lbls, size_t b_size, unsigned int seed = 42)
      : images(imgs), labels(lbls), batch_size(b_size), index(0), rng(seed)
  {
    if (images.size() != labels.size())
      throw std::runtime_error("Imágenes y etiquetas no coinciden en tamaño");

    indices.resize(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
  }

  bool has_next() const
  {
    return index < images.size();
  }

  void reset()
  {
    index = 0;
    std::shuffle(indices.begin(), indices.end(), rng);
  }

  std::pair<Tensor, std::vector<float>> next_batch()
  {
    size_t end = std::min(index + batch_size, images.size());
    size_t actual_batch_size = end - index;

    const int channels = images[0].shape[0];
    const int height = images[0].shape[1];
    const int width = images[0].shape[2];

    // Crear tensor de batch: [batch_size, C, H, W]
    std::vector<float> batch_data(actual_batch_size * channels * height * width);
    std::vector<float> batch_labels(actual_batch_size);

    for (size_t i = 0; i < actual_batch_size; ++i)
    {
      size_t idx = indices[index + i];
      const Tensor &img = images[idx];
      const std::vector<float> &data = img.data;

      std::copy(data.begin(), data.end(),
                batch_data.begin() + i * channels * height * width);

      batch_labels[i] = labels[idx];
    }

    Tensor batch_tensor({(int)actual_batch_size, channels, height, width}, batch_data);

    index = end;
    return {batch_tensor, batch_labels};
  }
};
