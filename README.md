<p align="center">
  # Artificial-Neural-Network-in-C
</p>

Simple neural network with two hidden layers based on 3b1b's course on Neural networks and Michael Nielsen's book on Networks and Deep Learning. Trained on MNIST database of handwritten digits with a hovering accuracy of 84% - 88% using sigmoid and softmax as activations and gradient descent with the goal of reducing cross entropy loss.

<img width="713" alt="Screenshot 2024-01-01 210555" src="https://github.com/MarcTy/Artificial-Neural-Network-in-C/assets/88467549/2fa9f867-5967-4503-84ae-4ce76bd1ba77">

Simple to compile, ensure that you compile on a linux shell with gcc and make installed. Run this command while in the folder:
```
make
```
Change directory into build folder and run program file:
```
./program
```
<p align="center">
  # Sidenotes
</p>

The number of hidden neurons can be modified and will significantly increase training time but also improve accuracy.

The initial cross entropy loss is a measure of the cross entropy loss at the start of training of one piece of input data (In this case a 28x28 input vector representing an MNIST digit) with a random initialization of normal distribution weights and biases.

The final cross entropy loss is a measure of the cross entropy loss of one piece of input data after the weights and biases have been trained.

Often times, the final cross entropy loss is observed to be higher than the initial cross entropy loss such as this example, this final cross entropy loss is an outlier as generally the model tends to predict accurately (Evident when the model is tested on test data). This irregular scenario albeit shows the model's inaccuracy and therefore accuracy itself would be a better representation of the the minimization of cross entropy loss (Wherein an increased accuracy is a lower cross entropy loss).

<img width="719" alt="Screenshot 2024-01-01 210638" src="https://github.com/MarcTy/Artificial-Neural-Network-in-C/assets/88467549/a70ded7d-55e6-420d-b643-18722d4ebddf">

