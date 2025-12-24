import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
data = load_breast_cancer()
x_raw = data.data
y = data.target
feature_names = data.feature_names

# Z-score Normalization
mean = np.mean(x_raw, axis=0)
std = np.std(x_raw, axis=0)

# Avoid division by zero in case of constant features
std[std == 0] = 1.0

x_normalized = (x_raw - mean) / std
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, x):
        # Mean centering
        self.mean = np.mean(x, axis=0)
        x_centered = x - self.mean

        # np.cov(x.T) expects rows as features, so we use transpose
        covariance_matrix = np.cov(x_centered.T)

        # Eigenvalue Decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors (descending)
        # Greater the eigenvalue the greater the variance captured
        indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[indices]
        sorted_eigenvectors = eigenvectors[:, indices]

        # Select top n_components
        self.components = sorted_eigenvectors[:, 0:self.n_components]

        # Calculate Explained Variance Ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = (self.eigenvalues[:self.n_components] / total_variance)

        return self

    def transform(self, x):
        # Project data
        return np.dot(x - self.mean, self.components)

    def inverse_transform(self, x_transformed):
        return np.dot(x_transformed, self.components.T) + self.mean

    def calculate_reconstruction_error(self, x):
        x_transformed = self.transform(x)
        x_reconstructed = self.inverse_transform(x_transformed)
        mean_squarred_error = np.mean((x - x_reconstructed) ** 2)
        return mean_squarred_error