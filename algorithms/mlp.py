import numpy as np
import logging
from .utils.weight_initialization import WeightInitializer
from .utils.activation_function import ActivationFunction

class MLP:

    def __init__(self, input_size, hidden_sizes, output_size,
                 hidden_layer_activation, output_activation = "softmax",
                 learning_rate=0.01, weight_initializer_method = 'random',
                 weight_initializer_distribution = 'normal',
                 loss_function = 'categorical_crossentropy'):
        
        #initialize model parameters
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layer_activation = hidden_layer_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.weight_initializer_method = weight_initializer_method
        self.weight_initalizer_distribution = weight_initializer_distribution

        #Initialize weights and biases for layers
        self.weights = []
        self.biases = []

        #Initialize caches for intermediate results for using in backpropagation
        self.layer_outputs = []

        #Initialize weights and biases for input_layer
        shape = (self.input_size, self.hidden_sizes[0])
        weight_matrix = WeightInitializer.initialize_weights(shape, self.weight_initializer_method, self.weight_initalizer_distribution)
        self.weights.append(weight_matrix)
        self.biases.append(np.zeros((1, self.hidden_sizes[0])))

        #Initialize weights and biases for hidden_layers
        for i in range(1, len(self.hidden_sizes)):
            shape = (self.hidden_sizes[i-1], self.hidden_sizes[i])
            weight_matrix = WeightInitializer.initialize_weights(shape, self.weight_initializer_method, self.weight_initalizer_distribution)
            self.weights.append(weight_matrix)
            self.biases.append(np.zeros((1, hidden_sizes[i])))

        #Initialize weights and biases for output_layers
        shape = (self.hidden_sizes[-1], self.output_size)
        weight_matrix = WeightInitializer.initialize_weights(shape, self.weight_initializer_method, self.weight_initalizer_distribution)
        self.weights.append(weight_matrix)
        self.biases.append(np.zeros((1, self.output_size)))


    def forward(self, x):
        self.layer_outputs = []
        self.layer_outputs.append(x)
        #Forward propagation through hidden layers
        layer_input = x
        for i in range(len(self.hidden_sizes)):
            activation_function = self.hidden_layer_activation[i]
            _input = np.dot(layer_input, self.weights[i]) + self.biases[i]
            layer_output = ActivationFunction.method_activation(_input, activation_function) 
            self.layer_outputs.append(layer_output)
            layer_input = layer_output
        
        #Forward propagation through output layer
        output_input = np.dot(layer_input, self.weights[-1]) + self.biases[-1]

        if self.output_activation in {'sigmoid', 'softmax', 'identity'}:
            output = ActivationFunction.method_activation(output_input, method=self.output_activation)
        else:
            raise ValueError('Output Activation Function can be any one of the following: sigmoid, softmax, or identity')

        self.layer_outputs.append(output)
        return output
    
    def backpropagation(self, x, y, y_pred):
        deltas = []

        #Backpropagation through output layer
        if self.loss_function == 'categorical_crossentropy':
            #Activation Function is mostly either sigmoid/sofmax. Derivative of loss function will be same as output_error 
            output_error = y - y_pred
            output_delta = output_error
        elif self.loss_function == 'mean_squared_error':
            output_error = (y - y_pred) #For simplicity, removing constants i.e., multipy with 2 and divide by N.
            activation_derivative = ActivationFunction.method_activation(y_pred, method=self.output_activation + '_derivative')
            output_delta = output_error*activation_derivative

        deltas.append(output_delta)
        #Backpropagation through hidden layers
        for i in range(len(self.hidden_sizes), 0, -1):
            layer_error = deltas[-1].dot(self.weights[i].T)
            activation_function = self.hidden_layer_activation[i-1]
            layer_delta = layer_error*ActivationFunction.method_activation(self.layer_outputs[i], activation_function + '_derivative')
            deltas.append(layer_delta)
        
        deltas.reverse()

        #Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += self.layer_outputs[i].T.dot(deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate


    def train(self, X, y, epochs, batch_size):
        num_samples = X.shape[0]

        for epoch in range(epochs):
            #shuffle the data
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, num_samples, batch_size):
                x_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = self.forward(x_batch)
                self.backpropagation(x_batch, y_batch, y_pred)

            if self.loss_function == "categorical_crossentropy":
                loss = -np.sum(y_batch * np.log(y_pred)) / len(y_batch)
            elif self.loss_function == "mean_squared_error":
                loss = np.mean(np.square(y_batch - y_pred))

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


    def predict(self, x):
        return self.forward(x)

