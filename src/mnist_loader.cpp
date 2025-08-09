#include "mnist_loader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>

static uint32_t read_uint32(std::ifstream &f) {
    uint32_t result = 0;
    f.read(reinterpret_cast<char*>(&result), 4);
    result = ((result & 0xFF) << 24) |
             ((result & 0xFF00) << 8) |
             ((result & 0xFF0000) >> 8) |
             ((result & 0xFF000000) >> 24);
    return result;
}

void load_mnist_images(const std::string &filename, Matrix &images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open " + filename);

    uint32_t magic = read_uint32(file);
    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file!");

    uint32_t num_images = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    images = Matrix(num_images, rows * cols);

    for (uint32_t i = 0; i < num_images; i++) {
        for (uint32_t j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images(i, j) = pixel / 255.0f; // normalize
        }
    }
}

void load_mnist_labels(const std::string &filename, std::vector<int> &labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open " + filename);

    uint32_t magic = read_uint32(file);
    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file!");

    uint32_t num_labels = read_uint32(file);
    labels.resize(num_labels);

    for (uint32_t i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }
}
