#include "activation.h"

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float out, float dval) {
    return out > 0 ? dval : 0;

}

Matrix ActivationReLU::forward(const Matrix& input) {
    output = input.apply(relu);
    return output;
}

Matrix ActivationReLU::backward(const Matrix& d_values) {
    return output.apply_with(d_values, relu_derivative);

}

