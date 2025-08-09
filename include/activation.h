#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"
#include <cmath>

class ActivationReLU {
public:
    Matrix output;
    Matrix dinputs;

    ActivationReLU() : output(Matrix(1, 1)), dinputs(Matrix(1, 1)) {}
    
    void forward(const Matrix &inputs);
    void backward(const Matrix &dvalues);
};

class ActivationSoftmax {
public:
    Matrix output;
    Matrix dinputs;

    ActivationSoftmax() : output(Matrix(1, 1)), dinputs(Matrix(1, 1)) {}

    void forward(const Matrix &inputs);
    void backward(const Matrix &dvalues);
};

#endif