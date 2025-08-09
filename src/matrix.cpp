#include "matrix.h"
#include <random>
#include <stdexcept>

Matrix::Matrix(int rows, int cols, bool zero) : rows(rows), cols(cols) {
    data.resize(rows * cols);
    if (zero) std::fill(data.begin(), data.end(), 0.0f);
}

Matrix Matrix::random(int rows, int cols, float min, float max) {
    Matrix m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);
    for (auto &val : m.data) val = dist(gen);
    return m;
}

float &Matrix::operator()(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::runtime_error("Matrix index out of bounds");
    }
    return data[r * cols + c];
}

float Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) {
        throw std::runtime_error("Matrix index out of bounds");
    }
    return data[r * cols + c];
}

Matrix Matrix::dot(const Matrix &a, const Matrix &b) {
    if (a.cols != b.rows) throw std::runtime_error("Matrix shape mismatch for dot product");
    Matrix result(a.rows, b.cols, true);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a.cols; k++)
                sum += a(i, k) * b(k, j);
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::transpose(const Matrix &m) {
    Matrix result(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result(j, i) = m(i, j);
    return result;
}

void Matrix::applyFunction(const std::function<float(float)> &func) {
    for (auto &val : data) val = func(val);
}

Matrix Matrix::subtract(const Matrix &a, const Matrix &b) {
    if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("Matrix size mismatch");
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.data.size(); i++)
        result.data[i] = a.data[i] - b.data[i];
    return result;
}

Matrix Matrix::multiplyElementWise(const Matrix &a, const Matrix &b) {
    if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("Matrix size mismatch");
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.data.size(); i++)
        result.data[i] = a.data[i] * b.data[i];
    return result;
}

Matrix Matrix::subtractScalar(const Matrix &a, float scalar) {
    Matrix result(a.rows, a.cols);
    for (int i = 0; i < a.data.size(); i++)
        result.data[i] = a.data[i] - scalar;
    return result;
}

void Matrix::add(const Matrix &other) {
    // Handle bias broadcasting: if other has 1 row and same cols, broadcast to all rows
    if (other.rows == 1 && cols == other.cols) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                (*this)(r, c) += other(0, c);
            }
        }
    }
    // Standard element-wise addition
    else if (rows == other.rows && cols == other.cols) {
        for (int i = 0; i < data.size(); i++) {
            data[i] += other.data[i];
        }
    }
    else {
        throw std::runtime_error("Matrix size mismatch for addition");
    }
}