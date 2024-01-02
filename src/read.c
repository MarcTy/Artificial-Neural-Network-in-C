#include "../headers/read.h"
#include "../headers/mat_ops.h"
#include "../headers/cond.h"
#include "../headers/net.h"
#include "../headers/layer.h"
#include "../headers/ops.h"

/*Functions for reading, training and testing model with data, aswell as memory allocs.*/

double *get_mem_double(size_t count)
{
    double *mem = calloc(count, sizeof(double));
    if (mem != NULL)
        return mem;
    else
    {
        fprintf(stderr, "\nFailed to allocate memory on one instance!\n");
        exit(1);
    }
}

activation_layers *get_mem_activation(void)
{
    activation_layers *mem = malloc(sizeof(activation_layers));
    if (mem != NULL)
        return mem;
    else
    {
        fprintf(stderr, "\nFailed to allocate memory on one instance!\n");
        exit(1);
    }
}

weight_and_bias_conditions *get_mem_conditions(void)
{
    weight_and_bias_conditions *mem = malloc(sizeof(weight_and_bias_conditions));
    if (mem != NULL)
        return mem;
    else
    {
        fprintf(stderr, "\nFailed to allocate memory on one instance!\n");
        exit(1);
    }
}

mat *get_mem_matrix(void)
{
    mat *mem = malloc(sizeof(mat));
    if (mem != NULL)
        return mem;
    else
    {
        fprintf(stderr, "\nFailed to allocate memory on one instance!\n");
        exit(1);
    }
}
/*Trains the bias and weights with the relevant training data.*/
void train_conditions(weight_and_bias_conditions *conditions, int batch_size)
{
    activation_layers *current_layers = get_mem_activation();
    layers_generate_initial_conditions(current_layers);
    weight_and_bias_conditions *changes_summation = get_mem_conditions();
    generate_zero_conditions(changes_summation);
    double initial_cross_entropy;
    double final_cross_entropy;

    FILE *file;
    file = fopen("../data/mnist_train.csv", "r");
    if (file == NULL)
    {
        fprintf(stderr, "Unable to open file\n");
        fclose(file);
        exit(1);
    }

    printf("\nCommencing training protocol");
    for (int i = 0; i < TRAINING_SIZE; i++)
    {
        /*Load visuals*/
        if ((i % (TRAINING_SIZE / 10)) == 0)
        {
            printf(".");
        }
        char line[2500];
        fgets(line, 2500, file);
        char *token = strtok(line, ",");
        int expected = (int)(token[0] - '0');
        double *buffer = get_mem_double(DIMENSIONS * DIMENSIONS);
        /*Read data line by line from file*/
        for (int j = 0; j < DIMENSIONS * DIMENSIONS; j++)
        {
            token = strtok(NULL, ",");
            buffer[j] = (double)(atoi(token) / 255.0);
        }
        mat *data_node = create_matrix(buffer, DIMENSIONS * DIMENSIONS, VEC_SIZE);

        if (i == 0)
        {
            /*Calculate initial cross entropy*/
            forward_feed(current_layers, conditions, data_node, expected);
            initial_cross_entropy = cross_entropy_loss(current_layers->output_layer, current_layers->expected_output_layer);
        }
        else if (((i % batch_size) == 0) && (i != 0))
        {
            /*Applies the changes/gradients to current bias and weights*/
            scale_contributions(changes_summation, (double)(LEARNING_RATE / batch_size));
            subtract_contributions(conditions, changes_summation);
            changes_summation = get_mem_conditions();
            generate_zero_conditions(changes_summation);
        }
        else
        {
            /*Sum up changes/gradients which will be averaged and applied to the weights and biases*/
            weight_and_bias_conditions *changes_to_conditions = get_mem_conditions();
            forward_feed(current_layers, conditions, data_node, expected);
            back_propagation(changes_to_conditions, current_layers, conditions, data_node);
            add_contributions(changes_summation, changes_to_conditions);
        }
        free_matrix(data_node);
    }
    /*Print out initial and final cross entropy loss*/
    final_cross_entropy = cross_entropy_loss(current_layers->output_layer, current_layers->expected_output_layer);
    printf("\nInitial Cross Entropy Loss: %f\nFinal Cross Entropy Loss: %f", initial_cross_entropy, final_cross_entropy);
    free_layers(current_layers);
    free_wbc(changes_summation);
    fclose(file);
}

/*Test the bias and weights with relevant test data.*/
void test_conditions(weight_and_bias_conditions *conditions)
{
    size_t correct_count = 0;
    size_t incorrect_count = 0;
    activation_layers *current_layers = get_mem_activation();
    layers_generate_initial_conditions(current_layers);
    FILE *file;
    file = fopen("../data/mnist_test.csv", "r");

    if (file == NULL)
    {
        fprintf(stderr, "Unable to open file\n");
        return;
    }

    printf("\n\nCommencing testing protocol");
    for (int i = 0; i < TEST_SIZE; i++)
    {
        if ((i % (TEST_SIZE / 10)) == 0)
        {
            printf(".");
        }
        char line[2500];
        fgets(line, 2500, file);

        char *token = strtok(line, ",");
        int expected = (int)(token[0] - '0');
        double *buffer = get_mem_double(DIMENSIONS * DIMENSIONS); /*Read data line by line from file*/
        for (int j = 0; j < DIMENSIONS * DIMENSIONS; j++)
        {
            token = strtok(NULL, ",");
            buffer[j] = (double)(atoi(token) / 255.0);
        }
        mat *data_node = create_matrix(buffer, DIMENSIONS * DIMENSIONS, VEC_SIZE);
        forward_feed(current_layers, conditions, data_node, expected);
        /*Test the model's output vector by finding the number with the highest probability.
        If the number with the highest probability is the expected number in question,
        the model has predicted correctly.*/
        int most_probable = 0;
        for (int k = 0; k < OUTPUT_SIZE; k++)
        {
            double *output = current_layers->output_layer->buff;
            if ((output[most_probable]) < (output[k]))
            {
                most_probable = k;
            }
        }
        /*Increment incorrect and correct counts*/
        if (expected == most_probable)
        {
            correct_count++;
        }
        else
        {
            incorrect_count++;
        }
        free_matrix(data_node);
    }
    printf("\nWithin %d test iterations, the model recieved an accuracy of %f%%, leaving %llu incorrect predictions.\n", TEST_SIZE, two_decimal((correct_count / (double)TEST_SIZE) * 100), incorrect_count);
    free_layers(current_layers);
    fclose(file);
}