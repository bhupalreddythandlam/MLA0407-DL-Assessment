import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 1. Toy Text Dataset
sentences = [
    "I love this amazing product", 
    "This is the best thing ever", 
    "Absolutely wonderful and fantastic",
    "I am very happy with this",
    "Great experience highly recommend",
    "Loved every second of it",
    "I hate this terrible product", 
    "This is the worst thing ever", 
    "Absolutely awful and disgusting",
    "I am very disappointed",
    "Terrible experience do not buy",
    "Hated every second of it"
]
labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 1 = Positive, 0 = Negative

# 2. Tokenize and Pad Sequences
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

max_len = 10
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# 3. Build RNN Model
model = Sequential([
    Embedding(input_dim=100, output_dim=16, input_length=max_len),
    LSTM(16),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training RNN for Text Classification...")
model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# Quick prediction test
test_text = ["This is amazing", "Terrible product"]
test_seq = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=max_len)
predictions = model.predict(test_seq)
print(f"Predictions (closer to 1 is positive): {predictions.flatten()}")