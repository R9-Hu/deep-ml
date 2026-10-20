import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	if not np.any(delta_W):
		return 0
	Sigma = np.linalg.svd(delta_W, compute_uv=False)
	ranks = Sigma.shape[0]
	summ = np.sum(Sigma**2)
	tmp = 0
	for k in range(ranks):
		tmp += Sigma[k]**2
		if tmp/summ >= energy_threshold:
			return k+1