#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i) {
        double sum = 0;
        for(j = 0; j < m.cols; ++j) {
            double x = m.data[i*m.cols + j];
            switch (a) {
            case LOGISTIC:
                x = 1 / (1 + exp(-x));
                break;
            case RELU:
                if (x <= 0) {
                    x = 0;
                }
                break;
            case LRELU:
                if (x <= 0) {
                    x = .01 * x;
                }
                break;
            case SOFTMAX:
                x = exp(x);
            }
            m.data[i * m.cols + j] = x;
            sum += x;
        }
        if (a == SOFTMAX) {
            for (j = 0; j < m.cols; j++) {
                m.data[i * m.cols + j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            // TODO: multiply the correct element of d by the gradient
            switch (a)
            {
            case LOGISTIC:
                d.data[1 * m.cols + j] *= x * (1-x);
            case RELU:
                if (x = 0) {
                    d.data[i * m.cols + j] = 0; // Grad is zero so multiply by zero
                }
                // Otherwise multiply by grad of 1.
            case LRELU:
                if (x < 0) {
                    d.data[i * m.cols + j] *= 0.01;
                }
                // Otherwise multiply by grad of 1.
            case SOFTMAX:
                // Always multiply by grad of 1.
                break;
        }
    }
}
