#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/*Mnist data specfic*/
#define TRAINING_SIZE 60000
#define TEST_SIZE 10000
#define DIMENSIONS 28
#define VEC_SIZE 1

/*The ith sample of training for when the weights and biases are updated.*/
#define BATCH_SIZE 20
/*Number of neurons in one hidden layer.*/
#define NO_OF_NEURONS 20
/*Size of the output layer.*/
#define OUTPUT_SIZE 10
/*Magnitude of gradient descent.*/
#define LEARNING_RATE 0.8
/*Amount of times trained on data.*/
#define NO_OF_EPOCHS 1