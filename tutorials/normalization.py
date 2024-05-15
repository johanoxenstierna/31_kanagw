import numpy as np

# Create a 2D numpy array with values between 0 and 1
arr = np.random.rand(3, 3)

# Normalize the values in the array to be between 0 and 1
arr_min = arr.min()
arr_max = arr.max()
normalized_arr = (arr - arr_min) / (arr_max - arr_min)

# Scale the normalized values to the new range of 2 to 4
new_min = 2
new_max = 4
modified_arr = normalized_arr * (new_max - new_min) + new_min

aa = 5
