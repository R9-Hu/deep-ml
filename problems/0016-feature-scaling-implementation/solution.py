import numpy as np
def feature_scaling(X: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = X.mean(axis=0, keepdims=True)
	std = X.std(axis=0, keepdims=True)
	minimum = X.min(axis=0, keepdims=True)
	span = X.max(axis=0, keepdims=True) - minimum
	standardized_data = np.round(
		(X - mean) / np.where(std == 0, 1.0, std), 4
	).tolist()
	normalized_data = np.round(
		(X - minimum) / np.where(span == 0, 1.0, span), 4
	).tolist()

	return standardized_data, normalized_data