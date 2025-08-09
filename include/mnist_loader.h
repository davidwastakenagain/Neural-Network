#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "matrix.h"
#include <string>
#include <vector>

void load_mnist_images(const std::string &filename, Matrix &images);
void load_mnist_labels(const std::string &filename, std::vector<int> &labels);

#endif
