#include "../headers/mat_ops.h"

/*Matrix calculations used in backpropagation calculation.*/

/*Matrices are represented as a structure containing a one dimensional array and integers that hold
column and row sizes. i.e 2D matrices are just 1D matrices of data that have a special way of interpretation.*/
mat *create_matrix(double *buff, int rows, int columns)
{
    mat *matrix = get_mem_matrix();
    matrix->buff = buff;
    matrix->rows = rows;
    matrix->columns = columns;
    return matrix;
}

/*Generates a zero matrix.*/
mat *generate_zero_matrix(int rows, int cols)
{
    double *buff = get_mem_double((size_t)(rows * cols));
    for (int i = 0; i < (rows * cols); i++)
    {
        buff[i] = 0;
    }
    return create_matrix(buff, rows, cols);
}

/*Generates a random matrix of number with normal distribution.*/
mat *generate_rand_matrix(int rows, int cols)
{
    double *buff = get_mem_double((size_t)(rows * cols));
    for (int i = 0; i < (rows * cols); i++)
    {
        buff[i] = sampleNormal();
    }
    return create_matrix(buff, rows, cols);
}

mat *matrix_multiplication(mat *a, mat *b)
{
    if (a->columns != b->rows)
    {
        fprintf(stderr, "Matrix multiplication failed, matrices cannot be multiplied");
    }

    int element_count = a->rows * b->columns;
    double *buff = get_mem_double((size_t)(element_count));
    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->columns; j++)
        {
            double sum = 0;
            for (int k = 0; k < a->columns; k++)
            {
                sum = sum + a->buff[i * a->columns + k] * b->buff[k * b->columns + j];
            }
            buff[i * b->columns + j] = sum;
        }
    }
    return create_matrix(buff, a->rows, b->columns);
}

mat *matrix_hadamard(mat *a, mat *b)
{
    if ((a->rows * a->columns) != (b->rows * b->columns))
    {
        fprintf(stderr, "Matrix hadamard failed, matrices are not the same size");
        return NULL;
    }
    int element_count = a->rows * a->columns;
    double *buff = get_mem_double((size_t)(element_count));
    for (int i = 0; i < element_count; i++)
    {
        buff[i] = a->buff[i] * b->buff[i];
    }
    return create_matrix(buff, a->rows, a->columns);
}

mat *matrix_addition(mat *a, mat *b)
{
    if ((a->rows * a->columns) != (b->rows * b->columns))
    {
        fprintf(stderr, "Matrix addition failed, matrices are not the same size");
        return NULL;
    }
    int element_count = a->rows * a->columns;
    double *buff = get_mem_double((size_t)(element_count));
    for (int i = 0; i < element_count; i++)
    {
        buff[i] = a->buff[i] + b->buff[i];
    }
    return create_matrix(buff, a->rows, a->columns);
}

mat *matrix_subtraction(mat *a, mat *b)
{
    if ((a->rows * a->columns) != (b->rows * b->columns))
    {
        fprintf(stderr, "Matrix subtraction failed, matrices are not the same size");
        return NULL;
    }
    int element_count = a->rows * a->columns;
    double *buff = get_mem_double((size_t)(element_count));
    for (int i = 0; i < element_count; i++)
    {
        buff[i] = a->buff[i] - b->buff[i];
    }
    return create_matrix(buff, a->rows, a->columns);
}

mat *matrix_scalar(mat *a, double scalar)
{
    int element_count = a->rows * a->columns;
    double *buff = get_mem_double((size_t)(element_count));
    for (int i = 0; i < element_count; i++)
    {
        buff[i] = a->buff[i] * scalar;
    }
    return create_matrix(buff, a->rows, a->columns);
}

mat *matrix_transpose(mat *a)
{
    int element_count = a->rows * a->columns;
    double *buff = get_mem_double((size_t)(element_count));
    for (int i = 0; i < element_count; i++)
    {
        buff[i] = a->buff[i];
    }
    return create_matrix(buff, a->columns, a->rows);
}

void normalize(mat *a)
{
    int element_count = a->rows * a->columns;
    double max = a->buff[0];
    double min = a->buff[0];
    for (int i = 0; i < element_count; i++)
    {
        if (a->buff[i] > max)
        {
            max = a->buff[i];
        }
        if (a->buff[i] < min)
        {
            min = a->buff[i];
        }
    }
    for (int i = 0; i < element_count; i++)
    {
        a->buff[i] = (a->buff[i] - min) / (max - min);
    }
}

void print_matrix(mat *a)
{
    for (int i = 0; i < (a->rows); i++)
    {
        for (int j = 0; j < (a->columns); j++)
        {
            printf("%f ", a->buff[i]);
        }
        printf("\n");
    }
    return;
}

void free_matrix(mat *a)
{
    free(a->buff);
    free(a);
    return;
}