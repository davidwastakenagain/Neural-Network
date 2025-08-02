#pragma once
#include "matrix.h"

class LayerDense {

public:
    Matrix weights, biases;
    Matrix output;
    Matrix d_weights, d_biases, d_inputs;
    Matrix input;

    LayerDense(int input_size, int output_size);
    Matrix forward(const Matrix& input);
    void backward(const Matrix& d_output);
};



