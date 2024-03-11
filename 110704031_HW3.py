# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y, weight=None):
    classes = np.unique(y)
    gini_impurity = 0
    
    for cls in classes:
        if weight is None:
            probabilities = len(y[y == cls]) / len(y)
        else:
            probabilities = np.sum(weight[y == cls]) / np.sum(weight)
        gini_impurity += probabilities * (1 - probabilities)
        
    return gini_impurity
    pass

# This function computes the entropy of a label array.
def entropy(y, weight=None):
    classes = np.unique(y)
    entropy_val = 0
    for cls in classes:
        if weight is None:
            probabilities = len(y[y == cls]) / len(y)
        else:
            probabilities = np.sum(weight[y == cls]) / np.sum(weight)
        entropy_val += probabilities * np.log2(probabilities)
    return -entropy_val
    pass

class Node:
    def __init__(self, feature_index=None, threshold=None, ineq=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.ineq = ineq
        self.value = value
        self.left = left
        self.right = right

       
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y, weight=None):
        if self.criterion == 'gini':
            return gini(y, weight)
        elif self.criterion == 'entropy':
            return entropy(y, weight)
    
    def fit(self, X, y, weight=None):
        self.tree = self._fit(X, y, depth=0, weight=weight)

    # This function fits the given data using the decision tree algorithm.
    def _fit(self, X, y, depth, weight):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # pure or reach max depth -> create leaf node
        if len(unique_classes) == 1 or depth == self.max_depth:
            return Node(value=np.argmax(np.bincount(y))) # 剩下的class label較多的
            

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y, weight)

        if best_feature is None:
            # No suitable split found, create a leaf node
            # print("no best feature")
            return Node(value=np.argmax(np.bincount(y)))

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        # Recursively build subtrees
        left_subtree = self._fit(X[left_mask], y[left_mask], depth + 1, weight)
        right_subtree = self._fit(X[right_mask], y[right_mask], depth + 1, weight)

        # Return the current node
        return Node(feature_index=best_feature, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)
        pass



    def _find_best_split(self, X, y, weight):
        num_samples, num_features = X.shape
        best_gain = -float('inf')
        best_feature, best_threshold = None, None

        for feature_index in range(num_features):  
            thresholds = np.unique(X[:, feature_index]) # feature內的所有值
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold
                # decision tree
                if weight is None:
                    info = self.impurity(y)
                    left_info = self.impurity(y[left_mask])
                    right_info = self.impurity(y[right_mask])
                # adaboost
                else:
                    info = self.impurity(y, weight)
                    left_info = self.impurity(y[left_mask], weight[left_mask])
                    right_info = self.impurity(y[right_mask], weight[right_mask])

                info_gain = info - ((len(y[left_mask]) / num_samples) * left_info + 
                                (1 - (len(y[left_mask]) / num_samples)) * right_info)


                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X, weight=None):
        return np.array([self._predict(sample, self.tree) for sample in X])
        pass
    
    def _predict(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict(sample, node.left)
        else:
            return self._predict(sample, node.right) 
        
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        feature_importance = np.zeros(len(columns))
        self._get_feature_importance(self.tree, feature_importance)
        # print(feature_importance)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(columns)), feature_importance)
        plt.yticks(range(len(columns)), columns)
        plt.xlabel('Feature Importance')
        plt.title('Decision Tree Feature Importance')
        plt.show()
        pass

    def _get_feature_importance(self, node, feature_importance):
        if node is not None:
            if node.feature_index is not None:
                feature_importance[node.feature_index] += 1
            self._get_feature_importance(node.left, feature_importance)
            self._get_feature_importance(node.right, feature_importance)

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []  # to store the weights of weak classifiers
        self.classifiers = []  # to store the weak classifiers
        self.weights = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_samples) / n_samples  # initialize sample weights

        for _ in range(self.n_estimators):
            # Create a weak classifier
            weak_classifier = DecisionTree(criterion=self.criterion, max_depth=1)
            weak_classifier.fit(X, y, self.weights)

            # Make predictions using the weak classifier
            y_pred = weak_classifier.predict(X)

            # Compute the error of the weak classifier
            err = np.sum(y_pred != y) / len(y)
            # print("error: ", err)

            # Avoid division by zero
            alpha = 0.5 * np.log((1 - err) / err)

            # Update sample weights
            self.weights[y_pred == y] *= np.exp(-alpha)
            self.weights[y_pred != y] *= np.exp(alpha)
            self.weights /= np.sum(self.weights)

            # Save the weak classifier and its weight
            self.classifiers.append(weak_classifier)
            self.alphas.append(alpha)
        

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        # Initialize the predicted labels
        y_pred = np.zeros(X.shape[0])

        # Aggregate predictions from all weak classifiers
        for weak_classifier, alpha in zip(self.classifiers, self.alphas):
            # weight只是用來判斷是否為adaboost
            # print("param: ", weak_classifier)
            pred = weak_classifier.predict(X)
            pred[pred==0] = -1
            # 用alpha投票
            y_pred += alpha * pred

        # Convert the weighted sum to class labels
        y_pred = np.sign(y_pred)
        y_pred[y_pred==-1] = 0
       
        return y_pred.astype(int)

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    # tree = DecisionTree(criterion='gini', max_depth=15)
    # tree.fit(X_train, y_train)
    # y_pred = tree.predict(X_test)
    # tree.plot_feature_importance_img(["age","sex","cp","fbs","thalach","thal"])

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=10)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))