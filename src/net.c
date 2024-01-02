#include "../headers/net.h"
#include "../headers/mat_ops.h"
#include "../headers/ops.h"
#include "../headers/layer.h"

/*Propagate forward with weights and bias passed into the second parameter forming
the layers of the network.*/
void forward_feed(activation_layers *layers, weight_and_bias_conditions *conditions, mat *data_node, int expected)
{
    /*One hot encoding*/
    double *buff = get_mem_double((size_t)10);
    buff[expected] = 1.0;
    mat *expected_output = create_matrix(buff, OUTPUT_SIZE, VEC_SIZE);
    free_matrix(layers->expected_output_layer);
    layers->expected_output_layer = expected_output;

    /*First hidden layer.*/
    mat *initial_to_hidden = layer_transition(conditions->weight.initial, data_node, conditions->bias.initial);
    mat *initial_layer = apply_sigmoid(initial_to_hidden);
    free_matrix(initial_to_hidden);
    free_matrix(layers->initial_layer);
    layers->initial_layer = initial_layer;

    /*Second hidden layer.*/
    mat *hidden_to_hidden = layer_transition(conditions->weight.hidden, initial_layer, conditions->bias.hidden);
    mat *hidden_layer = apply_sigmoid(hidden_to_hidden);
    free_matrix(hidden_to_hidden);
    free_matrix(layers->hidden_layer);
    layers->hidden_layer = hidden_layer;

    /*Final output layer.*/
    mat *hidden_to_output = layer_transition(conditions->weight.output, hidden_layer, conditions->bias.output);
    mat *output_layer = apply_softmax(hidden_to_output);
    free_matrix(hidden_to_output);
    free_matrix(layers->output_layer);
    layers->output_layer = output_layer;
}

/*Propogate backwards starting from the last layer in the network storing the gradients into the first parameter.*/
void back_propagation(weight_and_bias_conditions *changes, activation_layers *layers, weight_and_bias_conditions *conditions, mat *data_node)
{
    mat *combination_derivative;
    mat *transpose;
    mat *multi;

    /*Final layer transition.*/
    mat *cost_difference = matrix_subtraction(layers->output_layer, layers->expected_output_layer);
    changes->bias.output = cost_difference;
    transpose = matrix_transpose(layers->hidden_layer);
    mat *weight_derivatives = matrix_multiplication(cost_difference, transpose);
    free_matrix(transpose);
    changes->weight.output = weight_derivatives;

    /*Hidden layer transition.*/
    combination_derivative = apply_dsigmoid(layers->hidden_layer);
    transpose = matrix_transpose(conditions->weight.output);
    multi = matrix_multiplication(transpose, cost_difference);
    mat *cost_difference_two = matrix_hadamard(multi, combination_derivative);
    free_matrix(combination_derivative);
    free_matrix(multi);
    free_matrix(transpose);
    changes->bias.hidden = cost_difference_two;
    transpose = matrix_transpose(layers->initial_layer);
    mat *weight_derivatives_two = matrix_multiplication(cost_difference_two, transpose);
    free_matrix(transpose);
    changes->weight.hidden = weight_derivatives_two;

    /*First layer transition.*/
    combination_derivative = apply_dsigmoid(layers->initial_layer);
    transpose = matrix_transpose(conditions->weight.hidden);
    multi = matrix_multiplication(transpose, cost_difference_two);
    mat *cost_difference_three = matrix_hadamard(multi, combination_derivative);
    free_matrix(combination_derivative);
    free_matrix(multi);
    free_matrix(transpose);
    changes->bias.initial = cost_difference_three;
    transpose = matrix_transpose(data_node);
    mat *weight_derivatives_three = matrix_multiplication(cost_difference_three, transpose);
    free_matrix(transpose);
    changes->weight.initial = weight_derivatives_three;
}

/*Print network parameters.*/
void network_params(void)
{
    printf("~Network Parameters~\nInput Nodes: %d\nHidden Layers: 2\nHidden Nodes: %d\nOutput Nodes: %d\nHidden Layer Activation: Sigmoid\nOutput Layer Activation: Softmax\nLoss Function: Cross Entropy Loss\n",
           DIMENSIONS * DIMENSIONS, NO_OF_NEURONS, OUTPUT_SIZE);
}