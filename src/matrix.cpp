#include <iostream>
#include "matrix.h"

Matrix::Matrix(int r, int c) {
    rows = r;
    cols = c;
    data = new double[r * c]();
}

Matrix::~Matrix() {
    delete[] data;
}

void Matrix::print() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
