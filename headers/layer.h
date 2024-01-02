#ifndef LAYER
#define LAYER

#include "mat_ops.h"
#include "read.h"

void free_layers(activation_layers *layers);
void layers_generate_initial_conditions(activation_layers *a_layers);
mat *layer_transition(mat *weight_matrix, mat *activation_layer, mat *bias_vector);

#endif
