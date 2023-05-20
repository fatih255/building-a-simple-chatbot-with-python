from tensorflow import keras
import numpy as np

model = keras.models.Sequential()
model.add(keras.layers.Embedding(3,3))

input_array=np.array([0,1,2])
print(input_array)


output_array=model.predict(input_array)
print(output_array)
"""
print output:

input_array
[0 1 2]
output_array (Embedding)
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
 
"""
