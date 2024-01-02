#ifndef STRUCTS
#define STRUCTS

typedef struct matrix
{
    double *buff;
    int rows;
    int columns;
} mat;

typedef struct conditions
{
    mat *initial;
    mat *hidden;
    mat *output;
} cond;

typedef struct w_b_cond
{
    cond weight;
    cond bias;

} weight_and_bias_conditions;

typedef struct layer
{
    mat *initial_layer;
    mat *hidden_layer;
    mat *output_layer;
    mat *expected_output_layer;
} activation_layers;

#endif