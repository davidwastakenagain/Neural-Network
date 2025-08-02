#include "loss.h"
#include <cmath>
#include <cassert>
float LossMSE::forward(const Matrix& y_pred, const Matrix& y_true) {
    assert(y_pred.rows == y_true.rows && y_pred.cols == y_true.cols);
    float sum = 0.0;
    for (int i = 0; i < y_pred.rows; i++)
        for (int j = 0; j < y_pred.cols; j++) {
            float diff = y_pred.data[i][j] - y_true.data[i][j];
            sum += diff * diff;

        }

    return sum / (y_pred.rows * y_pred.cols);

}

Matrix LossMSE::backward(const Matrix& y_pred, const Matrix& y_true) {
    assert(y_pred.rows == y_true.rows && y_pred.cols == y_true.cols);
    return (y_pred - y_true) * (2.0f / (y_pred.rows * y_pred.cols));
}


