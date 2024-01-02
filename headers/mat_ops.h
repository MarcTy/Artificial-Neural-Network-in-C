#ifndef MATRIX_OPERATIONS
#define MATRIX_OPERATIONS

#include "structs.h"
#include "lib.h"
#include "read.h"

mat *create_matrix(double *buff, int rows, int columns);
mat *generate_rand_matrix(int rows, int cols);
mat *generate_zero_matrix(int rows, int cols);
mat *matrix_multiplication(mat *a, mat *b);
mat *matrix_hadamard(mat *a, mat *b);
mat *matrix_addition(mat *a, mat *b);
mat *matrix_subtraction(mat *a, mat *b);
mat *matrix_transpose(mat *a);
mat *matrix_scalar(mat *a, double scalar);
double sampleNormal(void);
void print_matrix(mat *a);
void free_matrix(mat *a);

#endif