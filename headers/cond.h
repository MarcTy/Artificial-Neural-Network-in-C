#ifndef COND
#define COND

#include "structs.h"
#include "lib.h"
#include "mat_ops.h"

void add_contributions(weight_and_bias_conditions *host, weight_and_bias_conditions *contributor);
void subtract_contributions(weight_and_bias_conditions *host, weight_and_bias_conditions *contributor);
void scale_contributions(weight_and_bias_conditions *host, double s);
void generate_initial_conditions(weight_and_bias_conditions *conditions);
void generate_zero_conditions(weight_and_bias_conditions *conditions);
void free_wbc(weight_and_bias_conditions *cond);

#endif