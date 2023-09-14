import numpy as np

class WeightInitializer:
    """
    A class for initializing weight matrices using various methods and distributions.

    Args:
    shape (tuple): The shape of the weight matrix to be initialized.
    method (str, optional): The initialization method to be used. Default is 'random'.
        Supported methods: 'random', 'xavier', 'he'.
    distribution (str, optional): The probability distribution to generate values from.
        Applicable only for 'random' initialization method. Default is 'normal'.
        Supported distributions: 'normal', 'uniform'.

    Returns:
    np.ndarray: The initialized weight matrix.

    Raises:
    ValueError: If an unsupported method is provided.

    Methods:
    - random_initialization(shape, distribution='normal'): Initialize weights using random values.
    - xavier_initialization(shape, distribution='normal'): Initialize weights using Xavier/Glorot initialization.
    - he_initialization(shape, distribution='normal'): Initialize weights using He initialization.
    """

    @staticmethod
    def initialize_weights(shape, method='random', distribution='normal'):
        """
        Apply the specified weight initialization method to the input data and return the result.

        Args:
        shape (tuple): The shape of the weight matrix to be initialized.
        method (str, optional): The initialization method to be used. Default is 'random'.
            Supported methods: 'random', 'xavier', 'he'.
        distribution (str, optional): The probability distribution to generate values from.
            Applicable only for 'random' initialization method. Default is 'normal'.
            Supported distributions: 'normal', 'uniform'.

        Returns:
        np.ndarray: The initialized weight matrix.

        Raises:
        ValueError: If an unsupported method is provided.
        """
        method_initializer = f"WeightInitializer.{method}_initialization({shape}, '{distribution}')"
        try:
            #print(method_initializer)
            weight_matrix = eval(method_initializer)
        except NameError:
            raise ValueError(f"Invalid method: {method}")
        return weight_matrix

    @staticmethod
    def random_initialization(shape, distribution='normal'):
        """
        Initialize weights with random values.

        Args:
        shape (tuple): The shape of the weight matrix to be initialized.
        distribution (str, optional): The probability distribution to generate values from.
            Default is 'normal'.
            Supported distributions: 'normal', 'uniform'.

        Returns:
        np.ndarray: The initialized weight matrix.
        """
        if distribution == 'normal':
            weight_matrix = np.random.normal(loc=0, scale=0.01, size=shape)
        else:
            weight_matrix = np.random.uniform(low=0, high=1, size=shape)
        return weight_matrix

    @staticmethod
    def xavier_initialization(shape, distribution='normal'):
        """
        Initialize weights using Xavier/Glorot initialization.

        Args:
        shape (tuple): The shape of the weight matrix to be initialized.
        distribution (str, optional): The probability distribution to generate values from.
            Default is 'normal'.
            Supported distributions: 'normal', 'uniform'.

        Returns:
        np.ndarray: The initialized weight matrix.
        """
        if distribution == 'normal':
            std_dev = np.sqrt(2 / (shape[0] + shape[1]))
            weight_matrix = np.random.normal(loc=0, scale=std_dev, size=shape)
        else:
            low = -np.sqrt(6) / np.sqrt(shape[0] + shape[1])
            high = np.sqrt(6) / np.sqrt(shape[0] + shape[1])
            weight_matrix = np.random.uniform(low=low, high=high, size=shape)
        return weight_matrix

    @staticmethod
    def he_initialization(shape, distribution='normal'):
        """
        Initialize weights using He initialization.

        Args:
        shape (tuple): The shape of the weight matrix to be initialized.
        distribution (str, optional): The probability distribution to generate values from.
            Default is 'normal'.
            Supported distributions: 'normal', 'uniform'.

        Returns:
        np.ndarray: The initialized weight matrix.
        """
        if distribution == 'normal':
            std_dev = np.sqrt(2 / shape[0])
            weight_matrix = np.random.normal(loc=0, scale=std_dev, size=shape)
        else:
            low = -np.sqrt(6) / np.sqrt(shape[0])
            high = np.sqrt(6) / np.sqrt(shape[0])
            weight_matrix = np.random.uniform(low=low, high=high, size=shape)
        return weight_matrix
