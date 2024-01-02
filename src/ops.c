#include "../headers/ops.h"
#include "../headers/mat_ops.h"

/*Mathematic operations used in forward and backpropagation on matrices, and etc.*/

/*Round to to two decimal places.*/
double two_decimal(double x) { return round(x * 100) / 100; }

/*Sigmoid.*/
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

/*Sigmoid derivative.*/
double dsigmoid(double x) { return (x * (1.0 - x)); }

/*Random number generator with normal distribution for initial weight calculation,
mean is zero, standard deviation is one.*/
double sampleNormal(void)
{
    double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1)
        return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

/*Calculate cross entropy loss for comparison.*/
double cross_entropy_loss(mat *o, mat *a)
{
    double log_sum = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        log_sum += (double)(a->buff[i] * log(o->buff[i]));
    }
    return -log_sum;
}

/*Softmax function to form the output layer.*/
mat *apply_softmax(mat *matrix)
{
    int element_count = matrix->rows * matrix->columns;
    double *buff = get_mem_double((size_t)element_count);
    double exp_summation = 0;
    for (int i = 0; i < (element_count); i++)
    {
        exp_summation += exp(matrix->buff[i]);
    }
    for (int j = 0; j < (element_count); j++)
    {
        buff[j] = exp(matrix->buff[j]) / exp_summation;
    }
    return create_matrix(buff, matrix->rows, matrix->columns);
}

/*Calculates the derivate of the softmax function as a jacobian matrix.
Not used in backpropagation currently, as the loss function is binary cross entropy
and there is no need.*/
mat *apply_dsoftmax(mat *matrix)
{
    mat *z = apply_softmax(matrix);
    mat *jacobian = generate_zero_matrix(matrix->rows, matrix->rows);
    int m = 0;
    int k = 0;
    for (int i = 0; i < (matrix->rows * matrix->rows); i++)
    {
        if (m == matrix->rows)
        {
            m = 0;
        }
        if (((i % matrix->rows) == 0) && (i != 0))
        {
            k++;
        }
        jacobian->buff[i] = -1 * (z->buff[m]) * (z->buff[k]);
        m++;
    }
    for (int j = 0; j < (matrix->rows); j++)
    {
        jacobian->buff[((matrix->rows) * j) + j] = (z->buff[j]) * (1 - z->buff[j]);
    }
    free_matrix(z);
    return jacobian;
}

/*Applies sigmoid to each value in a matrix/vector.*/
mat *apply_sigmoid(mat *matr)
{
    int element_count = matr->rows * matr->columns;
    double *buff = get_mem_double((size_t)element_count);

    for (int i = 0; i < (element_count); i++)
    {
        buff[i] = sigmoid(matr->buff[i]);
    }
    return create_matrix(buff, matr->rows, matr->columns);
}

/*Applies the derivative of the sigmoid to each value in a matrix/vector, specifically applied to
output vectors where the sigmoid has been applied to already.*/
mat *apply_dsigmoid(mat *matr)
{
    int element_count = matr->rows * matr->columns;
    double *buff = get_mem_double((size_t)element_count);
    for (int i = 0; i < (element_count); i++)
    {
        buff[i] = dsigmoid(matr->buff[i]);
    }
    return create_matrix(buff, matr->rows, matr->columns);
}
