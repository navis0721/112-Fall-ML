# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0

        # Gradient Descent
        for _ in range(self.iteration):
            linear_model = X @ self.weights + self.intercept
            pred = self.sigmoid(linear_model)

            # Compute gradients
            error = pred - y
            gradient_weights = X.T @ error / num_samples
            gradient_intercept = error.sum() / num_samples

            # Update weights and intercept
            self.weights -= self.learning_rate * gradient_weights
            self.intercept -= self.learning_rate * gradient_intercept
            # print("update weights: ", self.weights, self.intercept)
        # pass
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        pred = X @ self.weights + self.intercept
        probabilities = self.sigmoid(pred)
        # print("predicion: ", probabilities)
        return [1 if p > 0.5 else 0 for p in probabilities]
        # pass

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        # pass
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        # 分class0, class1的資料
        X0 = X[y == 0]
        X1 = X[y == 1]

        # 計算class mean
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        # sw = (Xn-m1)(Xn - m1)^T + (Xn-m2)(Xn - m2)^T
        self.sw = np.dot((X0 - self.m0).T, X0 - self.m0) + np.dot((X1 - self.m1).T, X1 - self.m1)

        # sb = (m2-m1)(m2-m1)^T
        self.sb = np.outer((self.m1 - self.m0), (self.m1 - self.m0))

        # w = sw^-1(m2-m1)
        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))
        # pass

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        # Project the data onto the FLD direction
        projection = np.dot(X, self.w)

        # Calculate the projection means
        projection_m0 = np.dot(self.m0, self.w)
        projection_m1 = np.dot(self.m1, self.w)

        # Predict class labels based on distances from the means
        y_pred = np.where(projection < (projection_m0 + projection_m1) / 2, 0, 1)

        return y_pred
        # pass

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        y_pred = self.predict(X)
        X0 = X[y_pred == 0]
        X1 = X[y_pred == 1]

        # Calculate the slope of the projection line
        avg = (np.mean(X0, axis=0) + np.mean(X1, axis=0)) / 2
        self.slope = self.w[1] / self.w[0]
        b = avg[1] - avg[0]*self.slope
        end0 = avg + 300000*self.w
        end1 = avg - 300000*self.w
        u0 = X0 - avg
        u1 = X1 - avg
        v = self.w
        project0 = (np.dot(u0, v) / np.dot(v, v))
        project1 = (np.dot(u1, v) / np.dot(v, v))
        # print(project0, project1)

        plt.figure(figsize=(6,14))
        plt.xlim(-10, 120)  # 假設 x 軸的範圍是 -10 到 120
        plt.ylim(80, 220)
        plt.title('Projection Line: w = {}, b = {}'.format(round(self.slope,2), round(b,2)))
        plt.scatter(X0[:, 0], X0[:, 1], c='b', marker='o', s=10, label='Class 0')
        plt.scatter(X1[:, 0], X1[:, 1], c='r', marker='o', s=10, label='Class 1')
        # Plot the projection line
        plt.plot([end0[0], end1[0]], [end0[1], end1[1]], 'k-', linewidth=1, label='Projection Line')
       
        for i in range (len(X0)):
            projection = np.dot(project0[i], v) + avg
            plt.plot([X0[i,0], projection[0]], [X0[i,1], projection[1]], color='gray', linestyle='-', linewidth=0.6, zorder=0)
            plt.scatter(projection[0], projection[1], c='b', marker='o', s=10)

        for i in range (len(X1)):
            projection = np.dot(project1[i], v) + avg
            plt.plot([X1[i,0], projection[0]], [X1[i,1], projection[1]], color='gray', linestyle='-', linewidth=0.6, zorder=0)
            plt.scatter(projection[0], projection[1], c='r', marker='o', s=10)

        plt.legend()
        plt.show()
        # pass
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.00045, iteration=20000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    FLD.plot_projection(X_test)
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

