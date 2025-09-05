import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#hvilke formler kan jeg bruke:
# z = SUM(w_i x_i + b, i = 1, ..., n) = wx + b
# sigmoid: sigma(z) = 1/(1 + e^-z)
# y_pred = 1/(1 + e -^(SUM(w_i * x_i +b), i = 1, ..., n))
# performance measure P: p(y|x) = y_pred^y (1 - y_pred)^(1-y) (gjelder hovedsakelig om y skal være binært)
# tapsfunksjon: L(y_pred, x) = -ln(p|x). modell defineres av w og b. må da finne parameterne som minimerer tapet i gjennomsnitt over alle datapunktene. mao: vil velge parameterne theta som maksimerer log-likelihood for target-verdiene y for hvert datapunkt x
# derivert av L mhp theta: dL/
# oppdatering av theta:
# theta(t+1) = theta(t) -n* dL(f(x;theta), y)/d theta
#  n: læringsrate 

class LinearRegression():
    
    def __init__(self, learning_rate = 0.1, epochs = 1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
    
    def _sigmoid(self, lin_model):
        s = 1/(1+ np.e**(-lin_model))
        return s

    def compute_gradients(self, x, y, y_pred):
        grad_w = (y_pred-y)*x
        grad_b = (y_pred-y)
        return grad_w, grad_b

    def compute_loss(self, y, y_pred):
        L = -y*np.log(y_pred)- (1-y)*np.log(1-y_pred)
        return L

    def update_parameters(self, grad_w, grad_b):
        for i in range (len(self.weights)):
            s = self.weights[i]-self.learning_rate*grad_w[i]
            print(s)
            self.weights[i] = s
        self.bias = self.bias - self.learning_rate*grad_b

    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
        
    def fit(self, x: list, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.weights = x #x.shape = [#samples, #features]
        self.bias = 0
        #Gradient Descent
        for _ in range (self.epochs):
            lin_model = np.matmul(self.weights, x) + self.bias
            y_pred = self._sigmoid(lin_model)
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self.compute_loss(y, y_pred)
            pred_to_class = 1 if y > 0.5 else 0 #problem child :(
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)
            
        

    
    def predict(self, x):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return [1 if y>0.5 else 0 for y in y_pred]




data = pd.read_csv('mission1.csv')

plt.figure(figsize=(6, 4))
plt.scatter(data['Net_Activity'], data['Energy'], c='blue', label='Data points')
plt.grid(True)
plt.xlabel('Network Activity', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.title('Energy vs. Traffic', fontsize=16)
plt.legend()
#plt.show()

lr = LinearRegression()

lr.fit(data['Net_Activity'], data['Energy'])

lr.predict(data['Net_Activity'])