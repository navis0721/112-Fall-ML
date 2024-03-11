# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):

        b=np.ones((X.shape[0], 1))
        X=np.hstack((X, b))
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.closed_form_weights = beta[:4]
        self.closed_form_intercept = beta[4]

        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        # pass

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):

        # b=np.ones((X.shape[0], 1))
        # X=np.hstack((X, b))
        coefficients = np.zeros(X.shape[1])
        intercept = 0.0

        mse = []

        # 開始梯度下降迭代
        for _ in range(epochs):
            # 計算預測值
            predictions = X @ coefficients + intercept
            mse.append(self.get_mse_loss(predictions, y))

            # 計算誤差（損失）
            error = predictions - y

            # 計算梯度
            gradient_coefficients = (2.0 / len(y)) * (X.T @ error)
            gradient_intercept = (2.0 / len(y)) * error.sum()

            # 更新模型參數
            coefficients -= lr * gradient_coefficients
            intercept -= lr * gradient_intercept

        self.gradient_descent_weights = coefficients
        self.gradient_descent_intercept = intercept
        # self.plot_learning_curve(epochs, mse)

        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        # pass
        

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):

        return np.square(prediction - ground_truth).mean()
        # Return the value.
        # pass

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):

        return X @ self.closed_form_weights + self.closed_form_intercept
        # Return the prediction.
        # pass

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):

        return X @ self.gradient_descent_weights + self.gradient_descent_intercept
        # Return the prediction.
        # pass
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self, epochs, mse):
        plt.figure(figsize=(16,3))
        plt.rcParams['figure.dpi'] = 227
        plt.plot(range(1, epochs + 1), np.array(mse))
        plt.title('Gradient Descent Optimization')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.show()
        # pass

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.00019, epochs=400000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
