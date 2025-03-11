import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def k_nn_predict(X_train, y_train, x_test, k=3):
    distances = [(euclidean_distance(x_test, x), y) for x, y in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

def evaluate_accuracy(X_train, y_train, X_test, y_test, k=3):
    correct = 0
    for i in range(len(X_test)):
        if k_nn_predict(X_train, y_train, X_test[i], k) == y_test[i]:
            correct += 1
    return correct / len(X_test)

def plot_decision_boundary(X, y, k=3):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = np.array([k_nn_predict(X, y, np.array([xx_i, yy_i]), k) for xx_i, yy_i in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f'k-NN Decision Boundary (k={k})')
    plt.show()

iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy = evaluate_accuracy(X_train, y_train, X_test, y_test, k=3)
print(f'Accuracy: {accuracy * 100:.2f}%')

plot_decision_boundary(X_train, y_train, k=3)
