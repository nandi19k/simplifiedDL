import numpy as np

class ActivationFunction:
    """
     A class for various activation functions used in neural networks.
    """
    @staticmethod
    def method_activation(input, method='sigmoid'):
        """
        Apply the specified activation method to the input data and return the result.

        Parameters:
        input (numpy.ndarray): Input data.
        method (str): The name of the activation method to be used. Default is 'sigmoid'.

        Returns:
        numpy.ndarray: Output after applying the activation.
        """
        method_code = f'ActivationFunction.{method}'
        try:
            result = eval(method_code)(input)
        except NameError:
            raise ValueError(f"Invalid method: {method}")
        return result

    @staticmethod
    def identity(x):
        """
        Identity activation function.
            f(x) = x

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying the Identity activation.
        """
        return x
    
    @staticmethod
    def identity_derivative(x):
        """
        Derivative of the identity activation function.
        f(x) = x

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Derivative of the identity activation.
        """
        # The derivative of the identity function is always 1.
        return np.ones_like(x)
    

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying the sigmoid activation.
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of the sigmoid activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Derivative of the sigmoid function.
        """
        return x * (1 - x)

    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying the ReLU activation.
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        Derivative of the ReLU activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        """
        Hyperbolic Tangent (tanh) activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying the tanh activation.
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """
        Derivative of the tanh activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Derivative of the tanh function.
        """
        return 1 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        """
        Softmax activation function.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Output after applying the softmax activation.
        """
        #subtract with max value in the array to have numerical stabiity
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
