#include "layer_dense.h"

LayerDense::LayerDense(int input_size, int output_size)
    : weights(Matrix::random(input_size, int output_size)),
    biases(Matrix(input_size, output_size)),
    d_weights(input_size, output_size),
    d_biases(1, output_size),
    d_inputs(0, 0) {}

Matrix LayerDense::forward(const Matrix& input_) {
    input = input_;
    output = input.dot(weights);
    return output;
}

void LayerDense::backward(const Matrix& d_output) {

    d_weights = input.transpose().dot(d_output);
    d_biases = d_output;
    d_inputs = d_output.dot(weights.transpose());
}
