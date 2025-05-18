#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <regex>
#include <string>
#include <iostream>

namespace fs = std::filesystem;

std::pair<std::vector<std::vector<float>>, std::vector<float>> load_dataset(const std::string& folder_path) {
    std::vector<std::vector<float>> X;
    std::vector<float> Y;
    std::regex label_regex(".*label_(\\d+)\\.png");  // Captura el número de la etiqueta

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::smatch match;

            if (std::regex_match(filename, match, label_regex)) {
                int label = std::stoi(match[1]);
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

                if (!img.empty()) {
                    // Aplanar la imagen a un vector de float
                    std::vector<float> flattened;
                    flattened.reserve(img.total());

                    for (int i = 0; i < img.rows; ++i) {
                        for (int j = 0; j < img.cols; ++j) {
                            flattened.push_back(static_cast<float>(img.at<uchar>(i, j)) / 255.0f); // Normalizado
                        }
                    }

                    X.push_back(flattened);
                    Y.push_back(static_cast<float>(label));
                } else {
                    std::cerr << "No se pudo cargar la imagen: " << entry.path() << std::endl;
                }
            }
        }
    }

    return {X, Y};
}
