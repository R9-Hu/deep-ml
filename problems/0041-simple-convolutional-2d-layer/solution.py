import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	input_matrix_pad = np.zeros((input_height+2*padding, input_width+2*padding))
	input_matrix_pad[padding:padding+input_height,padding:padding+input_width] = input_matrix
	output_height = int(np.floor((input_height+2*padding-kernel_height)/stride)+1)
	output_width = int(np.floor((input_width+2*padding-kernel_width)/stride)+1)
	output_matrix = np.zeros((output_height, output_width))
    
	for i in range(int(output_height)):
		for j in range(int(output_width)):
			output_matrix[i,j] = np.sum(input_matrix_pad[i*stride:i*stride+kernel_height,j*stride:j*stride+kernel_width]*kernel)
	return output_matrix
