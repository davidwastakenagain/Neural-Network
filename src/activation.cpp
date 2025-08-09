#include "activation.h"
#include <vector>
#include <cmath>


void ActivationReLU::forward(const Matrix &inputs) {
    output = inputs;
    for (auto &val : output.data)
        val = std::max(0.0f, val);
}

void ActivationReLU::backward(const Matrix &dvalues) {
    dinputs = Matrix(dvalues.rows, dvalues.cols);
    for (size_t i = 0; i < dinputs.data.size(); i++) {
        dinputs.data[i] = (output.data[i] > 0) ? dvalues.data[i] : 0.0f;
    }
}


void ActivationSoftmax::forward(const Matrix &inputs) {
    output = Matrix(inputs.rows, inputs.cols);
    for (int r = 0; r < inputs.rows; r++) {
        float max_val = -1e9;
        for (int c = 0; c < inputs.cols; c++)
            if (inputs(r, c) > max_val) max_val = inputs(r, c);

        float sum = 0.0f;
        for (int c = 0; c < inputs.cols; c++) {
            output(r, c) = std::exp(inputs(r, c) - max_val);
            sum += output(r, c);
        }
        for (int c = 0; c < inputs.cols; c++)
            output(r, c) /= sum;
    }
}

void ActivationSoftmax::backward(const Matrix &dvalues) {
    dinputs = dvalues;
}