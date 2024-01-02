#include "../headers/read.h"
#include "../headers/lib.h"
#include "../headers/net.h"
#include "../headers/cond.h"

/*ANN with two hidden layers batch trained on mnist data of handwritten digits with hovering accuracy between 81% - 88% over one epoch.
categorical cross entropy as the loss function with sigmoid activation for the hidden layers and softmax for the output layer. Forward
and backwards propagation calculations are done through matrix algebra with a simple matrix library written
for this project.*/

int main(void)
{
    srand(time(NULL));
    network_params();
    /*Struct container stores all weights and bias.*/
    weight_and_bias_conditions *conditions = get_mem_conditions();
    generate_initial_conditions(conditions);
    /*Train model based on number of epochs*/
    for (int i = 0; i < NO_OF_EPOCHS; i++)
    {
        printf("\nEpoch %d:", i + 1);
        train_conditions(conditions, BATCH_SIZE);
    }
    /*Test model with test file.*/
    test_conditions(conditions);
    free_wbc(conditions);
    return 0;
}