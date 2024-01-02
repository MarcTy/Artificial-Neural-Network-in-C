#include "../headers/cond.h"

/*Functions for modifiying the weight and bias conditions with the gradients and other functions related to modifying
the weights and bias struct.*/

/*Adds the contributor's matrices to the host parameter's, frees the contributor after computation.*/
void add_contributions(weight_and_bias_conditions *host, weight_and_bias_conditions *contributor)
{
    mat *add = matrix_addition(host->weight.initial, contributor->weight.initial);
    free_matrix(host->weight.initial);
    host->weight.initial = add;

    mat *add2 = matrix_addition(host->weight.hidden, contributor->weight.hidden);
    free_matrix(host->weight.hidden);
    host->weight.hidden = add2;

    mat *add3 = matrix_addition(host->weight.output, contributor->weight.output);
    free_matrix(host->weight.output);
    host->weight.output = add3;

    mat *add4 = matrix_addition(host->bias.initial, contributor->bias.initial);
    free_matrix(host->bias.initial);
    host->bias.initial = add4;

    mat *add5 = matrix_addition(host->bias.hidden, contributor->bias.hidden);
    free_matrix(host->bias.hidden);
    host->bias.hidden = add5;

    mat *add6 = matrix_addition(host->bias.output, contributor->bias.output);
    free_matrix(host->bias.output);
    host->bias.output = add6;
    free_wbc(contributor);
}

/*Adds the contributor's matrices to the host parameter's, frees the contributor after computation.*/
void subtract_contributions(weight_and_bias_conditions *host, weight_and_bias_conditions *contributor)
{
    mat *sub = matrix_subtraction(host->weight.initial, contributor->weight.initial);
    free_matrix(host->weight.initial);
    host->weight.initial = sub;

    mat *sub2 = matrix_subtraction(host->weight.hidden, contributor->weight.hidden);
    free_matrix(host->weight.hidden);
    host->weight.hidden = sub2;

    mat *sub3 = matrix_subtraction(host->weight.output, contributor->weight.output);
    free_matrix(host->weight.output);
    host->weight.output = sub3;

    mat *sub4 = matrix_subtraction(host->bias.initial, contributor->bias.initial);
    free_matrix(host->bias.initial);
    host->bias.initial = sub4;

    mat *sub5 = matrix_subtraction(host->bias.hidden, contributor->bias.hidden);
    free_matrix(host->bias.hidden);
    host->bias.hidden = sub5;

    mat *sub6 = matrix_subtraction(host->bias.output, contributor->bias.output);
    free_matrix(host->bias.output);
    host->bias.output = sub6;
    free_wbc(contributor);
}

/*Multiplies the passed in parameter's matrices by a scalar.*/
void scale_contributions(weight_and_bias_conditions *host, double s)
{
    mat *scale = matrix_scalar(host->weight.initial, s);
    free_matrix(host->weight.initial);
    host->weight.initial = scale;

    mat *scale2 = matrix_scalar(host->weight.hidden, s);
    free_matrix(host->weight.hidden);
    host->weight.hidden = scale2;

    mat *scale3 = matrix_scalar(host->weight.output, s);
    free_matrix(host->weight.output);
    host->weight.output = scale3;

    mat *scale4 = matrix_scalar(host->bias.initial, s);
    free_matrix(host->bias.initial);
    host->bias.initial = scale4;

    mat *scale5 = matrix_scalar(host->bias.hidden, s);
    free_matrix(host->bias.hidden);
    host->bias.hidden = scale5;

    mat *scale6 = matrix_scalar(host->bias.output, s);
    free_matrix(host->bias.output);
    host->bias.output = scale6;
}

/*Function that returns initial starting conditions of the weight matrixes, and bias vectors.*/
void generate_initial_conditions(weight_and_bias_conditions *conditions)
{
    conditions->weight.initial = generate_rand_matrix(NO_OF_NEURONS, DIMENSIONS * DIMENSIONS);
    conditions->weight.hidden = generate_rand_matrix(NO_OF_NEURONS, NO_OF_NEURONS);
    conditions->weight.output = generate_rand_matrix(OUTPUT_SIZE, NO_OF_NEURONS);

    conditions->bias.initial = generate_zero_matrix(NO_OF_NEURONS, VEC_SIZE);
    conditions->bias.hidden = generate_zero_matrix(NO_OF_NEURONS, VEC_SIZE);
    conditions->bias.output = generate_zero_matrix(OUTPUT_SIZE, VEC_SIZE);
}

void generate_zero_conditions(weight_and_bias_conditions *conditions)
{
    conditions->weight.initial = generate_zero_matrix(NO_OF_NEURONS, DIMENSIONS * DIMENSIONS);
    conditions->weight.hidden = generate_zero_matrix(NO_OF_NEURONS, NO_OF_NEURONS);
    conditions->weight.output = generate_zero_matrix(OUTPUT_SIZE, NO_OF_NEURONS);

    conditions->bias.initial = generate_zero_matrix(NO_OF_NEURONS, VEC_SIZE);
    conditions->bias.hidden = generate_zero_matrix(NO_OF_NEURONS, VEC_SIZE);
    conditions->bias.output = generate_zero_matrix(OUTPUT_SIZE, VEC_SIZE);
}

void free_wbc(weight_and_bias_conditions *c)
{
    free_matrix(c->bias.initial);
    free_matrix(c->bias.hidden);
    free_matrix(c->bias.output);
    free_matrix(c->weight.initial);
    free_matrix(c->weight.hidden);
    free_matrix(c->weight.output);
    free(c);
}