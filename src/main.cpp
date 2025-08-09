#include "matrix.h"
#include "layer_dense.h"
#include "activation.h"
#include "loss.h"
#include "mnist_loader.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

Matrix get_batch_data(const Matrix& X, const std::vector<int>& indices, int start, int batch_size) {
    int actual_batch_size = std::min(batch_size, (int)indices.size() - start);
    Matrix batch(actual_batch_size, X.cols);
    
    for (int i = 0; i < actual_batch_size; i++) {
        int idx = indices[start + i];
        for (int j = 0; j < X.cols; j++) {
            batch(i, j) = X(idx, j);
        }
    }
    return batch;
}

std::vector<int> get_batch_labels(const std::vector<int>& y, const std::vector<int>& indices, int start, int batch_size) {
    int actual_batch_size = std::min(batch_size, (int)indices.size() - start);
    std::vector<int> batch_labels(actual_batch_size);
    
    for (int i = 0; i < actual_batch_size; i++) {
        int idx = indices[start + i];
        batch_labels[i] = y[idx];
    }
    return batch_labels;
}

int main() {
    try {
        
        Matrix X_train(0, 0);
        std::vector<int> y_train;
        
        std::cout << "Loading MNIST data..." << std::endl;
        load_mnist_images("train-images.idx3-ubyte", X_train);
        load_mnist_labels("train-labels.idx1-ubyte", y_train);
        
        std::cout << "Loaded " << X_train.rows << " images with " << X_train.cols << " features each" << std::endl;

        LayerDense dense1(784, 128);
        ActivationReLU activation1;
        LayerDense dense2(128, 10);
        ActivationSoftmax activation2;

        float scale1 = std::sqrt(2.0f / 784.0f); 
        for (auto &w : dense1.weights.data) {
            w *= scale1;
        }
        
        float scale2 = std::sqrt(1.0f / 128.0f);
        for (auto &w : dense2.weights.data) {
            w *= scale2;
        }

        float learning_rate = 0.1f;
        int epochs = 5;
        int batch_size = 128;

        std::vector<int> indices(X_train.rows);
        for (int i = 0; i < X_train.rows; i++) {
            indices[i] = i;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Starting epoch " << (epoch + 1) << "..." << std::endl;

            std::shuffle(indices.begin(), indices.end(), gen);
            
            float epoch_loss = 0.0f;
            int epoch_correct = 0;
            int num_batches = 0;

            for (int start = 0; start < X_train.rows; start += batch_size) {
                Matrix X_batch = get_batch_data(X_train, indices, start, batch_size);
                std::vector<int> y_batch = get_batch_labels(y_train, indices, start, batch_size);

                dense1.forward(X_batch);
                activation1.forward(dense1.output);
                dense2.forward(activation1.output);
                activation2.forward(dense2.output);

                float batch_loss = 0.0f;
                int batch_correct = 0;

                for (int i = 0; i < X_batch.rows; i++) {
                    float confidence = std::max(activation2.output(i, y_batch[i]), 1e-7f);
                    batch_loss += -std::log(confidence);

                    int predicted = 0;
                    float max_val = activation2.output(i, 0);
                    for (int j = 1; j < 10; j++) {
                        if (activation2.output(i, j) > max_val) {
                            max_val = activation2.output(i, j);
                            predicted = j;
                        }
                    }
                    if (predicted == y_batch[i]) batch_correct++;
                }
                
                epoch_loss += batch_loss;
                epoch_correct += batch_correct;
                num_batches++;
                
                // Combined softmax+crossentropy backward pass
                Matrix softmax_grad = Matrix(activation2.output.rows, activation2.output.cols);
                for (int i = 0; i < activation2.output.rows; i++) {
                    for (int j = 0; j < activation2.output.cols; j++) {
                        softmax_grad(i, j) = activation2.output(i, j);
                        if (j == y_batch[i]) {
                            softmax_grad(i, j) -= 1.0f;
                        }
                    }
                }
                // Normalize by batch size
                for (auto &val : softmax_grad.data) {
                    val /= X_batch.rows;
                }

                dense2.backward(softmax_grad);
                activation1.backward(dense2.dinputs);
                dense1.backward(activation1.dinputs);

                for (size_t i = 0; i < dense1.weights.data.size(); i++) {
                    dense1.weights.data[i] -= learning_rate * dense1.dweights.data[i];
                }
                for (size_t i = 0; i < dense1.biases.data.size(); i++) {
                    dense1.biases.data[i] -= learning_rate * dense1.dbiases.data[i];
                }
                for (size_t i = 0; i < dense2.weights.data.size(); i++) {
                    dense2.weights.data[i] -= learning_rate * dense2.dweights.data[i];
                }
                for (size_t i = 0; i < dense2.biases.data.size(); i++) {
                    dense2.biases.data[i] -= learning_rate * dense2.dbiases.data[i];
                }

                if (num_batches % 100 == 0) {
                    float current_acc = (float)epoch_correct / (num_batches * batch_size);
                    float current_loss = epoch_loss / (num_batches * batch_size);
                    std::cout << "  Batch " << num_batches << " - Loss: " << current_loss 
                              << " - Accuracy: " << current_acc << std::endl;
                }
            }
            
            float final_loss = epoch_loss / X_train.rows;
            float final_accuracy = (float)epoch_correct / X_train.rows;
            
            std::cout << "Epoch " << (epoch+1)
                      << " - Loss: " << final_loss
                      << " - Accuracy: " << final_accuracy << std::endl;
        }

        std::cout << "Training completed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}