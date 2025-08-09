#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H

#include "matrix.h"

class LayerDense {
public:
    Matrix weights;
    Matrix biases;
    Matrix output;
    Matrix dweights;
    Matrix dbiases;
    Matrix input;
    Matrix dinputs;

    LayerDense(int n_inputs, int n_neurons);
    void forward(const Matrix &inputs);
    void backward(const Matrix &dvalues);
};

#endif
