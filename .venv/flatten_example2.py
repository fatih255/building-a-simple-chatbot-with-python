from tensorflow import keras

# Creating sample data
input_data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Creating the model
model = keras.models.Sequential()
model.add(keras.layers.Embedding(10, 5, input_length=3))

# The Flatten layer converts multi-dimensional data into a one-dimensional vector.
# This is necessary to perform operations in the subsequent fully connected (Dense) layers.
# The Flatten layer flattens the data, keeping the dimensions unchanged, into a single vector.

model.add(keras.layers.Flatten())

# The Dense layer is a fully connected artificial neural network layer.
# This layer takes a one-dimensional vector as input and produces outputs using a specific activation function.

model.add(keras.layers.Dense(10, activation='relu'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the data
model.fit(input_data, [0, 1, 0], epochs=10, batch_size=1)

# Making predictions on new data
new_data = [
    [2, 4, 6],
    [1, 3, 5]
]
predictions = model.predict(new_data)

print("Predictions:")
print(predictions)