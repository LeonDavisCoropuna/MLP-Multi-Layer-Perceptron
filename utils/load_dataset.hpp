#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <regex>
#include <string>
#include <iostream>
#include "tensor.hpp"

namespace fs = std::filesystem;
std::pair<std::vector<std::vector<float>>, std::vector<float>>
load_dataset_numbers(const std::string &folder_path, int max_samples = -1)
{
  std::vector<std::vector<float>> X;
  std::vector<float> Y;
  std::regex label_regex(".*?(\\d+)\\.png$");

  // 1. Recolectar archivos en un vector
  std::vector<fs::directory_entry> entries;
  for (const auto &entry : fs::directory_iterator(folder_path))
  {
    if (entry.is_regular_file())
    {
      entries.push_back(entry);
    }
  }

  // 2. Ordenar alfabéticamente por nombre de archivo
  std::sort(entries.begin(), entries.end(), [](const fs::directory_entry &a, const fs::directory_entry &b)
            { return a.path().filename() < b.path().filename(); });

  // 3. Procesar los archivos ordenados
  int count = 0;
  for (const auto &entry : entries)
  {
    if (max_samples != -1 && count >= max_samples)
      break;

    std::string filename = entry.path().filename().string();
    std::smatch match;

    if (std::regex_match(filename, match, label_regex))
    {
      int label = std::stoi(match[1]);
      cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

      if (!img.empty())
      {
        std::vector<float> flattened;
        flattened.reserve(img.total());

        for (int i = 0; i < img.rows; ++i)
        {
          for (int j = 0; j < img.cols; ++j)
          {
            flattened.push_back(static_cast<float>(img.at<uchar>(i, j)) / 255.0f);
          }
        }

        X.push_back(flattened);
        Y.push_back(static_cast<float>(label));
        ++count;
      }
      else
      {
        std::cerr << "No se pudo cargar la imagen: " << entry.path() << std::endl;
      }
    }
    else
    {
      std::cerr << "Nombre no coincide con regex: " << filename << std::endl;
    }
  }

  return {X, Y};
}

std::pair<std::vector<std::vector<float>>, std::vector<float>>
load_dataset_fashion(const std::string &csv_path)
{
  std::ifstream file(csv_path);
  std::string line;
  std::vector<std::vector<float>> images;
  std::vector<float> labels;

  // Leer encabezado y descartarlo
  std::getline(file, line);

  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string value;
    std::vector<float> pixels;

    // Leer la etiqueta
    std::getline(ss, value, ',');
    labels.push_back(std::stof(value));

    // Leer todos los píxeles
    while (std::getline(ss, value, ','))
    {
      pixels.push_back(std::stof(value) / 255.0f); // Normaliza los valores a [0, 1]
    }

    images.push_back(pixels);
  }

  return {images, labels};
}

std::vector<Tensor> loadImages2D(const std::string &filename, int max_images=9999999)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("No se pudo abrir el archivo de imágenes");

  int32_t magic = 0, num = 0, rows = 0, cols = 0;
  file.read(reinterpret_cast<char *>(&magic), 4);
  file.read(reinterpret_cast<char *>(&num), 4);
  file.read(reinterpret_cast<char *>(&rows), 4);
  file.read(reinterpret_cast<char *>(&cols), 4);

  magic = __builtin_bswap32(magic);
  num = __builtin_bswap32(num);
  rows = __builtin_bswap32(rows);
  cols = __builtin_bswap32(cols);

  if (max_images > 0 && max_images < num)
  {
    num = max_images;
  }

  std::vector<Tensor> images;
  images.reserve(num);

  for (int i = 0; i < num; ++i)
  {
    std::vector<float> data(rows * cols);
    for (int j = 0; j < rows * cols; ++j)
    {
      unsigned char pixel = 0;
      file.read(reinterpret_cast<char *>(&pixel), 1);
      data[j] = static_cast<float>(pixel) / 255.0f;
    }

    // Cada imagen tiene forma [1, rows, cols] → 1 canal
    Tensor image({1, rows, cols}, data);
    images.push_back(image);
  }

  return images;
}

std::vector<float> loadLabels(const std::string &filename, int max_labels=9999999)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("No se pudo abrir el archivo de etiquetas");

  int32_t magic = 0, num_labels = 0;
  file.read(reinterpret_cast<char *>(&magic), 4);
  file.read(reinterpret_cast<char *>(&num_labels), 4);

  magic = __builtin_bswap32(magic);
  num_labels = __builtin_bswap32(num_labels);

  if (max_labels > 0 && max_labels < num_labels)
  {
    num_labels = max_labels;
  }

  std::vector<float> labels(num_labels);
  for (int i = 0; i < num_labels; ++i)
  {
    unsigned char label = 0;
    file.read(reinterpret_cast<char *>(&label), 1);
    labels[i] = static_cast<int>(label);
  }

  return labels;
}
