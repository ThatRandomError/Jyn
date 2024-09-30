 Jyn Class Documentation
Overview
The Jyn class represents a neural network that can be trained using backpropagation. It manages the network's architecture, datasets, forward propagation, and gradient descent optimization.

Fields
ArrayList<Layer> architecture: The structure of the neural network, comprising its layers.
ArrayList<Integer> layers: A list of integers representing the size of each layer.
ArrayList<ArrayList<JMatrix>> dataset: The dataset used for training the neural network.
JMatrix output: The output produced by the network after forward propagation.
ArrayList<ArrayList<JMatrix>> nn: The list of weights and biases for the network.
ArrayList<ArrayList<JMatrix>> gradients: The calculated gradients for the weights and biases.
ArrayList<Function<JMatrix, JMatrix>> activation_functions: The activation functions for each layer.
ArrayList<Function<JMatrix, JMatrix>> activation_derv_functions: The derivatives of the activation functions for each layer.
Constructor
Jyn(ArrayList<Layer> architecture):
Parameters:
architecture: A list of Layer objects representing the architecture of the neural network.
Methods
void load_dataset(ArrayList<ArrayList<JMatrix>> dataset):

Description: Loads the training dataset into the network.
Parameters:
dataset: A list of input-target pairs for training.
double cost(JMatrix output, JMatrix target):

Description: Computes the cost (loss) between the predicted output and the target output.
Parameters:
output: The predicted output matrix.
target: The actual target matrix.
Returns: The computed cost.
void applyGradient(double learnRate):

Description: Updates the network weights and biases using the calculated gradients and the learning rate.
Parameters:
learnRate: The learning rate used for updating weights and biases.
ArrayList<JMatrix> activations(JMatrix inputs):

Description: Computes the activations for each layer given the input matrix.
Parameters:
inputs: The input matrix to the network.
Returns: A list of activation matrices for each layer.
void train(int epochs, double learnRate, boolean print):

Description: Trains the neural network using the provided dataset for a specified number of epochs.
Parameters:
epochs: The number of training iterations.
learnRate: The learning rate for gradient descent.
print: A boolean indicating whether to print the cost after each epoch.
JMatrix forward(JMatrix activations):

Description: Performs forward propagation through the network given the input activations.
Parameters:
activations: The input activations to the network.
Returns: The output of the network after forward propagation.
void init_weights():

Description: Initializes the weights and biases for the neural network randomly and sets up the gradient storage.
void save(String path):

Description: Saves the neural network's weights and biases to a file.
Parameters:
path: The file path where the network data should be saved.
void load(String path):

Description: Loads the neural network's weights and biases from a file.
Parameters:
path: The file path from which the network data should be loaded.
Layer Class Documentation
Overview
The Layer class represents a single layer in a neural network. Each layer consists of a specified number of neurons and has an associated activation function along with its derivative.

Fields
int size: The number of neurons in the layer.
Function<JMatrix, JMatrix> activation: The activation function used in the layer.
Function<JMatrix, JMatrix> derivative: The derivative of the activation function.
Constructor
Layer(int size, Function<JMatrix, JMatrix> activation, Function<JMatrix, JMatrix> derivative):
Parameters:
size: The number of neurons in the layer. Must be a positive integer.
activation: A function representing the activation function for the layer. If null, a default "none" function is used, which simply returns the input.
derivative: A function representing the derivative of the activation function. If null, a default "noneDerivative" function is used, which returns a zero matrix of the same dimensions as the input.
Throws: IllegalArgumentException if size is less than or equal to zero.
Methods
JMatrix none(JMatrix x):

Description: Default activation function that returns the input matrix without any changes.
Parameters:
x: The input matrix to be processed.
Returns: The same matrix x.
JMatrix noneDerivative(JMatrix x):

Description: Default derivative function that returns a zero matrix with the same dimensions as the input matrix.
Parameters:
x: The input matrix.
Returns: A zero matrix of the same dimensions as x.
Function<JMatrix, JMatrix> getActivation():

Description: Returns the activation function of the layer.
Returns: The activation function.
Function<JMatrix, JMatrix> getDerivative():

Description: Returns the derivative of the activation function.
Returns: The derivative function.
String toString():

Description: Returns a string representation of the layer, including its size, activation function, and derivative.
Returns: A string representation of the Layer object.
