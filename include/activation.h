
#pragma once
#include "matrix.h"

class ActivationReLU {

public:
    Matrix output;
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& d_values);

};


