#include "loss.h"
#include <cmath>
#include <stdexcept>

float LossCategoricalCrossentropy::forward(const Matrix &y_pred, const std::vector<int> &y_true) {
    int samples = y_pred.rows;
    std::vector<float> correct_confidences(samples);

    for (int i = 0; i < samples; i++) {
        float confidence = y_pred(i, y_true[i]);
        correct_confidences[i] = std::max(confidence, 1e-7f);
    }

    float sum_loss = 0.0f;
    for (float c : correct_confidences)
        sum_loss += -std::log(c);

    return sum_loss / samples;
}

void LossCategoricalCrossentropy::backward(const Matrix &dvalues, const std::vector<int> &y_true) {
    int samples = dvalues.rows;

    dinputs = Matrix(dvalues.rows, dvalues.cols, true);
    
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < dvalues.cols; j++) {
            dinputs(i, j) = dvalues(i, j);
        }
       
        dinputs(i, y_true[i]) -= 1.0f;
    }

    for (auto &val : dinputs.data) {
        val /= samples;
    }
}