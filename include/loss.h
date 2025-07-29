#pragma once
#include "matrix.h"

class LossMSE {
public:
    float forward(const Matrix& y_pred, const Matrix& y_true);
    Matrix backward(const Matrix& y_pred, const Matrix& y_true);


};