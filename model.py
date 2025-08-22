# model.py
import numpy as np
import pandas as pd

class SimpleNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_prior = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes:
            self.class_prior[c] = np.sum(y == c) / n_samples

        for feature in X.columns:
            self.feature_probs[feature] = {}
            if X[feature].dtype == 'object':
                for c in self.classes:
                    feature_values = X[feature][y == c]
                    self.feature_probs[feature][c] = feature_values.value_counts(normalize=True).to_dict()
            else:
                for c in self.classes:
                    feature_values = X[feature][y == c]
                    self.feature_probs[feature][c] = {
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values) + 1e-9
                    }

    def predict(self, X):
        predictions = []
        for _, sample in X.iterrows():
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.class_prior[c])
                likelihood = 0
                for feature in X.columns:
                    if X[feature].dtype == 'object':
                        if sample[feature] in self.feature_probs[feature][c]:
                            likelihood += np.log(self.feature_probs[feature][c][sample[feature]])
                        else:
                            likelihood += np.log(1e-9)
                    else:
                        mean = self.feature_probs[feature][c]['mean']
                        std = self.feature_probs[feature][c]['std']
                        exponent = -((sample[feature] - mean)**2 / (2 * std**2))
                        likelihood += exponent - np.log(np.sqrt(2 * np.pi * std**2))

                posteriors[c] = prior + likelihood

            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)