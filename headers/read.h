#ifndef READ
#define READ

#include "structs.h"
#include "lib.h"

void train_conditions(weight_and_bias_conditions *conditions, int batch_size);
void test_conditions(weight_and_bias_conditions *conditions);
weight_and_bias_conditions *get_mem_conditions(void);
activation_layers *get_mem_activation(void);
mat *get_mem_matrix(void);
double *get_mem_double(size_t count);

#endif
