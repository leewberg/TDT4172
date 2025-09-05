import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler



class LogisticRegression():


    def __init__(self, learning_rate = 0.1, epochs = 1000, degree = 10, treshold = 0.5):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.degree = degree
        self.treshold = treshold
        self.poly = PolynomialFeatures(degree = self.degree, include_bias=False)
        self.scaler = MinMaxScaler()
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []


    def _sigmoid(self, lin_model):
        s = 1/(1+ np.e**(-lin_model))
        return s


    def compute_gradients(self, x, y, y_pred, rows):
        grad_w = (1/rows) * np.dot(x.T, (y_pred-y))
        grad_b = (1/rows) * np.sum(y_pred-y)
        return grad_w, grad_b


    def compute_loss(self, y, y_pred):
        L = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return L


    def update_parameters(self, grad_w, grad_b):
        self.weights = self.weights-self.learning_rate*grad_w
        self.bias = self.bias - self.learning_rate*grad_b


    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)


    def fit(self, x: list, y):
        """
        Estimates parameters for the classifier

                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        X_poly = self.poly.fit_transform(x)
        X_poly = self.scaler.fit_transform(X_poly)
        rows, cols = X_poly.shape
        self.weights = np.zeros(cols) #x.shape = [#samples, #features]
        self.bias = 0
        #Gradient Descent
        for _ in range (self.epochs):
            lin_model = np.dot(X_poly, self.weights) + self.bias
            y_pred = self._sigmoid(lin_model)

            grad_w, grad_b = self.compute_gradients(X_poly, y, y_pred, rows)
            self.update_parameters(grad_w, grad_b)


            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            pred_to_class = np.where(y_pred > self.treshold, 1, 0)
            self.train_accuracies.append(self.accuracy(y, pred_to_class))

    def predict(self, x):
        """
        Generates predictions
        
        Returns:
            A length m array of floats
        """
        X_poly = self.poly.transform(x)
        X_poly = self.scaler.transform(X_poly)

        lin_model = np.dot(X_poly, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return np.where(y_pred>self.treshold, 1, 0)