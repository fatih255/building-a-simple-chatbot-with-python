import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords
from fuzzywuzzy import fuzz

# Load the CSV file with "|" delimiter using the 'python' engine
faq_data = pd.read_csv('data/faq_data_for_saas_company.csv',
                       delimiter='|', engine='python')

# Data Preprocessing: Lowercasing, removing stopwords, and extra spaces


def preprocess_text(text):
    words = text.lower().strip().split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


faq_data['Question'] = faq_data['Question'].apply(preprocess_text)
faq_data['Response'] = faq_data['Response'].str.lower().str.strip()

# Convert questions to a dictionary for exact matching
faq_dict = dict(zip(faq_data['Question'], faq_data['Response']))

# Extract questions and responses for model training
train_data = faq_data['Question'].tolist()
train_labels = faq_data['Response'].tolist()

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_labels)

# Tokenize the text data
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences)

# Load pretrained GloVe embeddings


def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Update this path to where your GloVe file is located
glove_path = 'data/glove/glove.6B.100d.txt'
embedding_dim = 100
embedding_matrix = load_glove_embeddings(
    glove_path, tokenizer.word_index, embedding_dim)

# Build the model with pretrained embeddings and a Bidirectional LSTM
model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                 output_dim=embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=train_sequences.shape[1],
                                 trainable=False))
model.add(keras.layers.Bidirectional(
    keras.layers.LSTM(128, return_sequences=True)))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(
    len(np.unique(encoded_labels)), activation='softmax'))

# Compile the model with a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with validation split
model.fit(train_sequences, encoded_labels, epochs=100, batch_size=32,
          validation_split=0.3, callbacks=[early_stopping])

# Prepare TF-IDF vectorizer for similarity checking
vectorizer = TfidfVectorizer().fit(train_data)


def fuzzy_match(query, faq_dict):
    best_match = None
    highest_ratio = 0
    for question, response in faq_dict.items():
        ratio = fuzz.ratio(query, question)
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = response
    return best_match if highest_ratio > 70 else None  # Adjust threshold as needed


def generate_response(text):
    # Preprocess input
    text = preprocess_text(text)

    # Exact or fuzzy matching
    if text in faq_dict:
        return faq_dict[text]
    else:
        fuzzy_response = fuzzy_match(text, faq_dict)
        if fuzzy_response:
            return fuzzy_response

    # Use the model if exact or fuzzy match is not found
    sequence = tokenizer.texts_to_sequences([text])
    sequence = keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=train_sequences.shape[1])

    # Predict and decode the label
    prediction = model.predict(sequence)
    predicted_label = np.argmax(prediction)
    response = label_encoder.inverse_transform([predicted_label])[0]

    # Check similarity
    input_vector = vectorizer.transform([text])
    similarities = cosine_similarity(
        input_vector, vectorizer.transform(train_data))
    max_similarity = similarities.max()

    # Return a fallback response if similarity is below a threshold
    if max_similarity < 0.7:
        return "Sorry, I cannot answer that question, it is not in my data."

    return response


# Chat loop
while True:
    user_input = input("Enter a message: ")
    response = generate_response(user_input)
    print("ChatBot: ", response)