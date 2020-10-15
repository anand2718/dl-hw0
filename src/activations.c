#define LEAKY_RATE 0.1
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
            double f_x = m.data[i*m.cols + j];
            switch (a) {
            case LINEAR:
                break;
            case LOGISTIC:
                f_x = 1 / (1 + exp(-f_x));
                break;
            case RELU:
                if (f_x <= 0) {
                    f_x = 0;
                }
                break;
            case LRELU:
                if (f_x <= 0) {
                    f_x = LEAKY_RATE * f_x;
                }
                break;
            case SOFTMAX:
                f_x = exp(f_x);
                break;
            }
            m.data[i * m.cols + j] = f_x;
            sum += f_x;
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
            double f_x = m.data[i*m.cols + j];
            // TODO: multiply the correct element of d by the gradient
            switch (a)
            {
            case LINEAR:
                // Always multiply by grad of 1.
                break;
            case LOGISTIC:
                d.data[i * m.cols + j] *= (f_x * (1-f_x));
                break;
            case RELU:
                if (f_x <= 0) {
                    d.data[i * m.cols + j] = 0; // Grad is zero so multiply by zero
                }
                break;
                // Otherwise multiply by grad of 1.
            case LRELU:
                if (f_x < 0) {
                    d.data[i * m.cols + j] *= LEAKY_RATE;
                }
                break;
                // Otherwise multiply by grad of 1.
            case SOFTMAX:
                // Always multiply by grad of 1.
                break;
            }
        }
    }
}
