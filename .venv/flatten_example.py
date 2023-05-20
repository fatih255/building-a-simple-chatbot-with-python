from tensorflow import keras
import numpy as np

input_array = np.array([[0, 1, 2], [3, 4, 5]])
print("Input Array:")
print(input_array)

model_with_flatten = keras.models.Sequential()
model_with_flatten.add(keras.layers.Embedding(6, 3, input_length=3))
model_with_flatten.add(keras.layers.Flatten())

model_without_flatten = keras.models.Sequential()
model_without_flatten.add(keras.layers.Embedding(6, 3, input_length=3))

output_array_with_flatten = model_with_flatten.predict(input_array)
output_array_without_flatten = model_without_flatten.predict(input_array)

print("\nOutput Array with Flatten:")
print(output_array_with_flatten)

print("\nOutput Array without Flatten:")
print(output_array_without_flatten)


"""
print output:

Output Array with Flatten:
[[ 0.04758482 -0.00341137  0.00157311 -0.03944163 -0.00405199 -0.01363243
  -0.04762021  0.02439376  0.00533192]
 [ 0.01037518 -0.01396211 -0.03730686  0.03879153  0.00464141 -0.04617056
   0.03882095  0.00657393  0.02819261]]

Output Array without Flatten:
[[[-0.04261672  0.02249751  0.03048887]
  [-0.04016012  0.01420603 -0.04359663]
  [ 0.02580397 -0.0371437  -0.04547833]]

 [[ 0.01448012 -0.03714436 -0.04263431]
  [ 0.0450354  -0.04823824  0.01479555]
  [-0.01878364 -0.01148184  0.00181563]]]
"""
