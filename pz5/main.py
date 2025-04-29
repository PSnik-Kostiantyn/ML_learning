import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

k = 3
np.random.seed(42)
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

def closest_centroid(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def move_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

for iteration in range(100):
    labels = closest_centroid(X, centroids)
    new_centroids = move_centroids(X, labels, k)
    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

def pca(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    top_vectors = eig_vecs[:, sorted_indices[:n_components]]
    return X_centered @ top_vectors

X_pca = pca(X, n_components=2)
centroids_pca = pca(centroids, n_components=2)

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(k):
    plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], c=colors[i], label=f'Кластер {i+1}')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=250, c='yellow', marker='X', label='Центроїди')
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.title('Кластеризація k-середніх (4 ознаки, PCA візуалізація)')
plt.legend()
plt.grid(True)
plt.show()
