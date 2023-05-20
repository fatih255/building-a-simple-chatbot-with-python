from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np


train_data = [
    "Hello",
    "How are you?",
    "Good morning",
    "Good evening",
    "Nice to meet you",
    "What's up?",
    "How's your day going?",
    "Greetings!",
    "Good afternoon",
    "How can I assist you?",
    "Pleasure to see you",
    "Is there anything I can help with?"
]

train_labels = [
    "Hi",
    "I'm fine, how about you?",
    "Good morning to you",
    "Good evening, how can I help you?",
    "Nice to meet you too",
    "Not much, just hanging out",
    "It's going well, thank you",
    "Hello!",
    "Good afternoon to you too",
    "I'm here to assist you",
    "Likewise!",
    "Yes, I have a question"
]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_labels)

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences)

model = keras.models.Sequential()
model.add(keras.layers.Embedding(len(tokenizer.word_index) + 1, 100, input_length=train_sequences.shape[1]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(train_labels), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_sequences, encoded_labels, epochs=50)

def generate_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=train_sequences.shape[1])
    prediction = model.predict(sequence)
    predicted_label = np.argmax(prediction)
    response = label_encoder.inverse_transform([predicted_label])[0]
    return response

while True:
    user_input = input("Enter a message: ")
    response = generate_response(user_input)
    print("ChatBot: ", response)
