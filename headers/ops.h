#ifndef OPS
#define OPS

#include "structs.h"
#include "lib.h"

double sigmoid(double x);
double dsigmoid(double x);
double two_decimal(double x);
double cross_entropy_loss(mat *o, mat *a);
mat *apply_sigmoid(mat *matrix);
mat *apply_dsigmoid(mat *matrix);
mat *apply_softmax(mat *matrix);
mat *apply_dsoftmax(mat *matrix);

#endif
