#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>

class Matrix {
public:
    int rows, cols;
    std::vector<float> data;

    Matrix(int rows, int cols, bool zero = false);
    static Matrix random(int rows, int cols, float min = -1.0f, float max = 1.0f);
    float &operator()(int r, int c);
    float operator()(int r, int c) const;

    static Matrix dot(const Matrix &a, const Matrix &b);
    static Matrix transpose(const Matrix &m);
    void applyFunction(const std::function<float(float)> &func);
    static Matrix subtract(const Matrix &a, const Matrix &b);
    static Matrix multiplyElementWise(const Matrix &a, const Matrix &b);
    static Matrix subtractScalar(const Matrix &a, float scalar);
    void add(const Matrix &other);
};

#endif
