
import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.000001, epochs=10000, threshold=0.5):
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.losses = []
        self.threshold = threshold
        self.train_accuracies = []

    def _sigmoid(self, X):
        return 1 / ( 1 + np.exp(-X))
    
    def _linear_model(self, X):
        return np.dot(X, self.weights) + self.bias

    def _loss_function(self, y, y_pred):
        return np.mean(-y * np.log(y_pred) - (1-y) * np.log(1 - y_pred))

    def _compute_gradients(self, X, y, y_pred):
        grad_w = (1 / len(X)) * np.dot(X.T, (y_pred - y))
        grad_b = np.mean(y_pred - y)
        return grad_w, grad_b

    def _update_parameters(self, grad_w, grad_b):
        self.bias -= self.learning_rate * grad_b
        self.weights -= self.learning_rate * grad_w

    def _accuracy(self, truth, pred):
        return np.mean(truth == pred)

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
            y_pred = self._sigmoid(self._linear_model(X))
            grad_w, grad_b = self._compute_gradients(X, y, y_pred)
            self._update_parameters(grad_w, grad_b)

            loss = self._loss_function(y, y_pred)
            y_pred_to_class = [1 if _y >= self.threshold else 0 for _y in y_pred]
            self.train_accuracies.append(self._accuracy(y, y_pred_to_class))
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
        y_pred = self._sigmoid(self._linear_model(X))
        return [1 if _y >= self.threshold else 0 for _y in y_pred]




