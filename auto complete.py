import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
import math

# Load Sherlock Holmes Dataset 
with open('C:\\Users\\pc\\OneDrive\\Desktop\\projects\\next word generator\\Sherlock Holmes.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Preprocess the text (tokenization, cleaning, etc.)
corpus = text.splitlines()  # Split by lines or any delimiter that suits your dataset

# Tokenize the corpus
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # Include 0 for padding

# Prepare input sequences and labels
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Split into predictors (X) and label (y)
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

class PositionalEncoding(Layer):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def call(self, inputs):
        sequence_length = tf.shape(inputs)[1]  # Get the sequence length dynamically
        position = tf.range(sequence_length, dtype=tf.float32)  # Create a range of positions
        position = tf.expand_dims(position, axis=1)  # Shape: (sequence_length, 1)

        # Compute the division term for sine and cosine
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))

        # Compute sine and cosine positional encodings
        even_indices = tf.sin(position * div_term)  # Shape: (sequence_length, d_model/2)
        odd_indices = tf.cos(position * div_term)  # Shape: (sequence_length, d_model/2)

        # Concatenate sine and cosine encodings along the last axis
        pos_encoding = tf.concat([even_indices, odd_indices], axis=-1)  # Shape: (sequence_length, d_model)

        # Add a batch dimension to match the input shape
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Shape: (1, sequence_length, d_model)

        return inputs + pos_encoding



# Multi-Head Attention Layer
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0  # d_model must be divisible by num_heads
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        return output, attention_weights

# Transformer Block (same as your original code)
class TransformerBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([ 
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, mask=None):
        attn_output, _ = self.mha(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Transformer Model
class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, sequence_length, d_model, num_heads, ff_dim, num_blocks, lstm_units):
        super(TransformerModel, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_blocks)]
        self.lstm = LSTM(lstm_units, return_sequences=False)
        self.dropout = Dropout(0.2)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, inputs, mask=None):
        x = self.embedding(inputs)
        x = self.positional_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)  # Remove mask here if not needed

        x = self.lstm(x)
        x = self.dropout(x)
        return self.dense(x)

# Model parameters
d_model = 128  # Dimensionality of the embedding space
num_heads = 8  # Number of attention heads
ff_dim = 512  # Feed-forward network size
num_blocks = 4  # Number of transformer blocks
lstm_units = 128  # Number of units in the LSTM layer

# Instantiate the model
model = TransformerModel(vocab_size=total_words, sequence_length=max_sequence_length, 
                         d_model=d_model, num_heads=num_heads, ff_dim=ff_dim,
                         num_blocks=num_blocks, lstm_units=lstm_units)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64)

# Function to generate next word
def generate_next_word(seed_text, model, tokenizer, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted_prob = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_prob, axis=-1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    return predicted_word

# Example: Predict the next word after a seed text
seed_text = "how are you "
predicted_word = generate_next_word(seed_text, model, tokenizer, max_sequence_length)
print(f"Predicted next word: {predicted_word}")
