import numpy as np
from sklearn.linear_model import LinearRegression

class ModelPackage:
    
    def predict(train_data, test_data):
        model = LinearRegression()

        X_train = train_data[:, :-1]   
        y_train = train_data[:, -1]   

        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred, y_test
    
    def split_dataset(data, test_size=0.2, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        
        shuffled_data = data[np.random.permutation(len(data))]
        
        test_count = int(len(data) * test_size)
        
        test_data = shuffled_data[:test_count]
        train_data = shuffled_data[test_count:]
        
        return train_data, test_data

    class SimpleLinearRegression:
        
        def __init__(self):
            self.slope = None
            self.intercept = None

        def simple_linear_regression(self, train_data):
            
            x_train = train_data[:, 0]
            y_train = train_data[:, 1]
            
            x_mean = np.mean(x_train)
            y_mean = np.mean(y_train)
            
            numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
            denominator = np.sum((x_train - x_mean) ** 2)
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            self.slope = slope
            self.intercept = intercept
            
            return slope, intercept

        def simple_predict(self, test_data):
            x_test = test_data[:, 0]
            y_test = test_data[:, 1]

            y_pred = self.slope * x_test + self.intercept
            
            return y_pred, y_test
        

    class MultipleLinearRegression:
        
        def __init__(self):
            self.coefficients = None
            self.slopes = None
            self.intercept = None

        def multiple_linear_regression(self, train_data):
        
            x_train = train_data[:, :-1]
            y_train = train_data[:, -1]
            
            x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
            
            coefficients = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
            intercept = coefficients[0]
            slopes = coefficients[1:]
            
            self.coefficients = coefficients
            self.slopes = slopes
            self.intercept = intercept
            return slopes, intercept

        def multiple_predict(self, test_data):
            x_test = test_data[:, :-1]
            y_test = test_data[:, -1]

            y_pred = self.intercept + np.sum(self.slopes * x_test, axis=1)
            
            return y_pred, y_test

