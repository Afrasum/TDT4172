import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.000001, epochs=10000):
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.losses = []
    
    def _linear_model(self, X):
        return np.dot(X, self.weights) + self.bias

    def _MSE(self, y, y_pred):
        N = len(y)
        return (1 / (2 * N)) * np.sum((y - y_pred) ** 2)
        
    def _compute_gradients(self, X, y, y_pred):
        N = len(y_pred)
        grad_w = (1 / N) * np.dot(X.T, (y_pred - y))
        grad_b = np.mean(y_pred - y)
        return grad_w, grad_b

    def _update_parameters(self, grad_w, grad_b):
        self.bias -= self.learning_rate * grad_b
        self.weights -= self.learning_rate * grad_w

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = self._linear_model(X)
            grad_w, grad_b = self._compute_gradients(X, y, y_pred)
            self._update_parameters(grad_w, grad_b)

            loss = self._MSE(y, y_pred)
            self.losses.append(loss)


    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return self._linear_model(X)





