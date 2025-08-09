#include "layer_dense.h"
#include <random>

LayerDense::LayerDense(int n_inputs, int n_neurons)
    : weights(Matrix::random(n_inputs, n_neurons, -0.1f, 0.1f)),  
      biases(Matrix(1, n_neurons, true)),
      output(Matrix(1, 1)),
      dweights(Matrix(1, 1)),
      dbiases(Matrix(1, 1)),
      input(Matrix(1, 1)),
      dinputs(Matrix(1, 1)) {}

void LayerDense::forward(const Matrix &inputs) {
    input = inputs;
    output = Matrix::dot(inputs, weights);
    
    for (int r = 0; r < output.rows; r++) {
        for (int c = 0; c < output.cols; c++) {
            output(r, c) += biases(0, c);
        }
    }
}

void LayerDense::backward(const Matrix &dvalues) {
    dweights = Matrix::dot(Matrix::transpose(input), dvalues);
    dbiases = Matrix(1, dvalues.cols, true);
    
   
    for (int c = 0; c < dvalues.cols; c++) {
        float sum = 0.0f;
        for (int r = 0; r < dvalues.rows; r++)
            sum += dvalues(r, c);
        dbiases(0, c) = sum;
    }
    dinputs = Matrix::dot(dvalues, Matrix::transpose(weights));
}