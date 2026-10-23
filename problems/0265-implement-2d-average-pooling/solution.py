import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	# Your code here
	input_matrix_np = np.asarray(input_matrix)
	m, n = input_matrix_np.shape
	out_np = np.zeros((m//pool_size, n//pool_size))
	for i in range(m//pool_size):
		for j in range(n//pool_size):
			out_np[i,j] = np.mean(input_matrix_np[i*pool_size:(i+1)*pool_size,j*pool_size:(j+1)*pool_size])
	return out_np.tolist()