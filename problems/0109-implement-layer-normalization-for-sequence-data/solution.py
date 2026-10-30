import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	batch_size, seq_len, d_model = X.shape
	mean_val = X.mean(axis=2, keepdims=True)
	var_val = X.var(axis=2, keepdims=True)
	X_norm = (X-mean_val)/np.sqrt(var_val+epsilon)
	return gamma*X_norm+beta
