#include <iostream>
#include "matrix.h"
#include "layer_dense.h"
#include "activation.h"
#include "loss.h"

int main() {
    std::cout << "Neural net starting..."<< std::endl;

    Matrix X = Matrix::random(2,3);
    std::cout <<"Input:\n";
    X.print();

    LayerDense layer1(3,5);
    Matrix output1 = layer1.forward(X);
    std::cout << "Output of layer 1:\n";
    output1.print();

    ActivationReLU activation1;
    Matrix activated = activation1.forward(output1);
    std::cout << "Output after activation:\n";
    activated.print();

    Matrix y_true = Matrix::random(2,5);
    std::cout << "y_true:\n";
    y_true.print();

    LossMSE loss_function;
    float loss = loss_function.forward(activated, y_true);
    std::cout << "Loss:" << loss << std::endl;

    Matrix d_loss = loss_function.backward(activated, y_true);
    Matrix d_activation = activation1.backward(d_loss);
    layer1.backward(d_activation);

    float learning_rate = 0.01f;
    layer1.weights = layer1.weights - layer1.d_weights * learning_rate;
    layer1.biases = layer1.biases - layer1.d_biases * learning_rate;

    return 0;


}
