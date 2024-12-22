import numpy as np
import cloudpickle as pickle
from scipy.spatial.distance import cdist
import time

# Implementasi Gaussian Naive-Bayes from scratch
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        self.epsilon = 1e-9

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + self.epsilon
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        posteriors = []
        for x in X:
            posteriors_class = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c])) - 0.5 * np.sum(((x - self.mean[c]) ** 2) / (self.var[c]))
                posteriors_class.append(prior + likelihood)
            posteriors.append(self.classes[np.argmax(posteriors_class)])
        return np.array(posteriors)
        