import numpy as np
def feature_scaling(X: np.ndarray) -> (np.ndarray, np.ndarray):
	epsilon = 1e-6
	standardized_data = np.round((X-X.mean(axis=0,keepdims=True))/np.sqrt(X.var(axis=0,keepdims=True)+epsilon),4).tolist()
	normalized_data = np.round((X-X.min(axis=0,keepdims=True))/(X.max(axis=0,keepdims=True)-X.min(axis=0,keepdims=True)+epsilon),4).tolist()
	return standardized_data, normalized_data