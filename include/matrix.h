#ifndef MATRIX_H
#define MATRIX_H

struct Matrix {
    int rows;
    int cols;
    double* data;

    Matrix(int r, int c);
    ~Matrix();

    void print();
};

#endif
