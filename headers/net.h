#ifndef NET
#define NET

#include "structs.h"
#include "lib.h"

void forward_feed(activation_layers *layers, weight_and_bias_conditions *conditions, mat *data_node, int expected);
void back_propagation(weight_and_bias_conditions *changes, activation_layers *layers, weight_and_bias_conditions *conditions, mat *data_node);
void network_params(void);

#endif