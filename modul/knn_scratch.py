import numpy as np
import cloudpickle as pickle
from scipy.spatial.distance import cdist
import time

# Implementasi KNN from scratch
class KNN:
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = cdist(X, self.X_train, metric=self.metric)
        neighbors_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbors_labels = self.y_train[neighbors_idx]
        predictions = np.array([np.argmax(np.bincount(neigh)) for neigh in neighbors_labels])
        return predictions

    def predict_batch(self, X, batch_size=1000):
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            predictions.extend(self.predict(batch))
        return np.array(predictions)
