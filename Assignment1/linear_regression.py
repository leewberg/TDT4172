import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression():
    
    def __init__(self, learning_rate = 0.1, epochs = 1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias, self.X_min, self.X_maks = None, None, None, None
        self.losses, self.train_accuracies = [], []
    
    def normalize(self, X):
        if isinstance(X, pd.Series):
            X = X.values
        else:
            X=X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        else:
            X=X

        X = (X - self.X_min)/(self.X_maks - self.X_min)
        return X
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.X_min = X.min(axis = 0)
        self.X_maks = X.max(axis=0)
        X = self.normalize(X)
        rows, cols = X.shape
        self.weights = np.zeros(cols)
        self.bias = 0
        #Gradient Descent
        for _ in range (self.epochs):
            lin_model = np.dot(X, self.weights)+self.bias
            diff_w = (1/rows) *np.dot(X.T, (lin_model-y))
            diff_b = (1/rows) * np.sum(lin_model - y)
            self.weights = self.weights - self.learning_rate*diff_w
            self.bias = self.bias - self.learning_rate*diff_b            
        

    
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
        X = self.normalize(X)
        lin_model = np.dot(X, self.weights) + self.bias
        return lin_model