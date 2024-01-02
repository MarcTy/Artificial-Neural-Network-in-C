#include "../headers/layer.h"

/*Layer transition for forward propagation.*/
mat *layer_transition(mat *weight_matrix, mat *activation_layer, mat *bias_vector)
{
    mat *multi = matrix_multiplication(weight_matrix, activation_layer);
    mat *add = matrix_addition(multi, bias_vector);
    free_matrix(multi);
    return add;
}

/*Used to predefine activation layers for initial null checking of free in forward propagation.*/
void layers_generate_initial_conditions(activation_layers *a_layers)
{
    a_layers->initial_layer = create_matrix(get_mem_double((size_t)1), 0, 0);
    a_layers->hidden_layer = create_matrix(get_mem_double((size_t)1), 0, 0);
    a_layers->output_layer = create_matrix(get_mem_double((size_t)1), 0, 0);
    a_layers->expected_output_layer = create_matrix(get_mem_double((size_t)1), 0, 0);
}

void free_layers(activation_layers *layers)
{
    free_matrix(layers->initial_layer);
    free_matrix(layers->hidden_layer);
    free_matrix(layers->output_layer);
    free_matrix(layers->expected_output_layer);
    free(layers);
}