#include "matrix.h"
#include <iostream>
#include <random>

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<float>(c,0)) {}

Matrix Matrix::random(int rows, int cols) {
    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    for (auto& row : result.data)
        for (auto& val : row)
                val = d(gen);

    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < other.cols; j++)
            for (int k = 0; k < cols; k++)
                result.data[i][j] += data[i][k] * other.data[k][j];
    
    return result;
}

Matrix Matrix::transpose() const {

    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j< cols; j++)
            result.data[j][i] = data[i][j];

    return result;
}
Matrix Matrix::apply(float (*func)(float)) const { 
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = func(data[i][j]);

    return result;
}

Matrix Matrix::apply_with(const Matrix& other, float (*func)(float, float)) const {

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = func(data[i][j], other.data[i][j]);

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j <cols; j++)
            result.data[i][j] = data[i][j] - other.data[i][j];

    return result;
}

Matrix Matrix::operator*(float scalar) const {

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j; j < cols; j++)
            result.data[i][j] = data[i][j] * scalar;

    return result;
}

void Matrix::print() const {
    for (const auto& row : data) {

        for (float val : row)
            std::cout <<val << "";
        std::cout << std::endl;
    }
}