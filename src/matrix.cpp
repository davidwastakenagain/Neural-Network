#include "matrix.h"
#include <iostream>
#include <random>
#include <cassert>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<float>(c, 0.0f)) {}

Matrix Matrix::random(int rows, int cols) {
    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = static_cast<float>(dis(gen));

    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    assert(cols == other.rows);
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < other.cols; ++j)
            for (int k = 0; k < cols; ++k)
                result.data[i][j] += data[i][k] * other.data[k][j];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
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
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = func(data[i][j], other.data[i][j]);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    assert(rows == other.rows && cols == other.cols);
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] - other.data[i][j];
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows == other.rows && cols == other.cols) {
        
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    if (other.rows == 1 && other.cols == cols) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[0][j];
        return result;
    }

    throw std::runtime_error("Matrix operation shape mismatched");

}
Matrix Matrix::operator*(float scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result.data[i][j] = data[i][j] * scalar;
    return result;
}

Matrix Matrix::sum_rows() const {
    Matrix result(1, cols);
    for (int j = 0; j < cols; ++j) {
        float sum = 0;
        for (int i = 0; i < rows; ++i)
            sum += data[i][j];
        result.data[0][j] = sum;
    }
    return result;
}

void Matrix::print() const {
    for (const auto& row : data) {
        for (float val : row)
            std::cout << val << " ";
        std::cout << std::endl;
    }
}
