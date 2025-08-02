#pragma once
#include <vector>
#include <functional>

class Matrix {

public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    Matrix();
    Matrix(int rows, int cols);
    static Matrix random(int rows, int cols);
   
    Matrix dot(const Matrix& other) const;
    Matrix transpose() const;
    Matrix apply(float (*func)(float)) const;
    Matrix apply_with(const Matrix& other, float (*func)(float, float)) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(float scalar) const;
    Matrix sum_rows() const;
    void print() const;

  

};
