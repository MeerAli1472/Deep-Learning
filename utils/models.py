import numpy as np
import os
import joblib


class Perceptron:
    def __init__(self,eta: float=None, epochs: int=None):  #eta ---> Learning rate
        self.weights = np.random.randn(3) * 1e-4  #small random weights, Why 3 becuase we have two columns x1,x2 and bias
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"initail weights before training: \n {self.weights}")
        self.eta = eta
        self.epochs = epochs
    
    def _z_outcome(self, x_with_bias, weights):
        return np.dot(x_with_bias, weights)
    
    def activationfunction(self,z):
        return np.where(z > 0, 1, 0) # if z greter then zero return 1 else 0
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        x_with_bias = np.c_[self.X, -np.ones((len(self.X),1))]
        print(f"X with bias: \n {x_with_bias}")
        
        for epoch in range(self.epochs):
            print(f"for epoch >> {epoch}")
            
            z = self._z_outcome(x_with_bias,self.weights)
            y_hat = self.activationfunction(z)
            print(f"predicted value after farward pass {y_hat}")
            
            self.error = y - y_hat
            print(f"Error: \n {self.error}")
            
            #Updates The weight
            self.weights = self.weights + self.eta * np.dot(x_with_bias.T,self.error)
            print(f"Updated weights after epoch: {epoch + 1}/{self.epochs}:\n{self.weights}")
            print("##"*10)
            
            
    
    def predict(self,X): # Here X is a test input
        x_with_bias = np.c_[X, -np.ones((len(X),1))]
        z = self._z_outcome(x_with_bias,self.weights)
        return self.activationfunction(z)
    
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total loss: {total_loss}")
        return total_loss
    
    def _create_dir_return_path(self,model_dir,filename):
        os.makedirs(model_dir,exist_ok=True)
        return os.path.join(model_dir,filename)
    
    
    def save(self,filename,model_dir=None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir,filename)
            joblib.dump(self,model_file_path)
        else:
            model_file_path = self._create_dir_return_path("model",filename)
            joblib.dump(self,model_file_path)
    
    def load(self,filepath):
        return joblib.load(filepath)
    
        