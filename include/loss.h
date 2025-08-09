#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include <vector>

class Loss {
public:
    virtual float forward(const Matrix &y_pred, const std::vector<int> &y_true) = 0;
    virtual void backward(const Matrix &dvalues, const std::vector<int> &y_true) = 0;
    virtual ~Loss() = default;
};

class LossCategoricalCrossentropy : public Loss {
public:
    Matrix dinputs;
    
    LossCategoricalCrossentropy() : dinputs(Matrix(1, 1)) {}
    
    float forward(const Matrix &y_pred, const std::vector<int> &y_true) override;
    void backward(const Matrix &dvalues, const std::vector<int> &y_true) override;
};

#endif