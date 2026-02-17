print("Script started...")
import os
# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=2, 
    restore_best_weights=True
)
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=1, 
    min_lr=1e-6,
    verbose=1
)

# loading dataset
df=pd.read_csv('cleaned_twitter_data.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
MODEL_FILE='sentiment_model.keras'
TOKENIZER_FILE='tokenizer.pkl'


# 1.text tokenization and preprocessing
#parameters
vocab_size=16000
max_length=100

#initialize and fit the tokenizer
tokenizer=Tokenizer(num_words=vocab_size,oov_token='<OOV>')
tokenizer.fit_on_texts(df['review_text'])

# transform text to sequences of numbers
sequences = tokenizer.texts_to_sequences(df['review_text'])
X_padded = pad_sequences(sequences, maxlen=max_length, padding='pre', truncating='post')


# 1. Load GloVe into a dictionary
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# 2. Create the matrix for your specific vocabulary
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # Words not found in GloVe will be random (helps generalization)
            embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

# 2. preparing the the multi-head labels
# sentiment labels
encoder=LabelEncoder()
y_sentiment=encoder.fit_transform(df['sentiment_label'])

# intensity labels
analyzer=SentimentIntensityAnalyzer()
def get_internsiy(text):
    score=analyzer.polarity_scores(str(text))
    return abs(score['compound'])

y_intensity=df['review_text'].apply(get_internsiy).values


# 3. building the multi-task learning model
# Shared Input and Embedding Layer
inputs= Input(shape=(max_length,),name="text_input")
# embedding=Embedding(vocab_size,128)(inputs)
# Replace your current embedding line with this:
embedding = Embedding(
    vocab_size, 
    embedding_dim, 
    weights=[embedding_matrix], 
    trainable=True, # This is key for generalization! 
    name="glove_embedding"
)(inputs)
spartial_dropout=SpatialDropout1D(0.5)(embedding)

# Shared LSTM Layer
# return_sequences=False summarizes the whole review into one vector
shared_lstm = Bidirectional(LSTM(64,return_sequences=True,dropout=0.5, kernel_regularizer=l2(0.0001)))(spartial_dropout)
shared_lstm = Bidirectional(LSTM(32,return_sequences=False,dropout=0.5, kernel_regularizer=l2(0.0001)))(shared_lstm)

# hidden layer
# --- Head 1: Sentiment Classification (3 units for Pos, Neu, Neg) ---
sentiment_dense = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(shared_lstm)
sentiment_dense = Dropout(0.5)(sentiment_dense)
# --- Head 2: Intensity Prediction (1 unit for 0.0 to 1.0) ---
intensity_dense = Dense(16, activation='relu')(shared_lstm)
intensity_dense = Dropout(0.2)(intensity_dense)
intensity_dense = Dense(8, activation='relu')(intensity_dense)

# output layer
sentiment_out = Dense(3, activation='softmax', name='sentiment_head')(sentiment_dense)
intensity_out = Dense(1, activation='sigmoid', name='intensity_head')(intensity_dense)

# Define the Model
model = Model(inputs=inputs, outputs=[sentiment_out, intensity_out])

# 4.compiling and training
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss={
        'sentiment_head':'sparse_categorical_crossentropy',
        'intensity_head':'mse'
    },
    loss_weights={
        'sentiment_head': 1.0,  # Focus primarily on getting the category right
        'intensity_head': 0.5   # Focus secondary on the intensity strength
    },
    metrics={'sentiment_head': 'accuracy', 'intensity_head': 'mse'}
)

# 5.training
model.fit(X_padded,
          { 'sentiment_head':y_sentiment,'intensity_head':y_intensity},
          epochs=10,
          batch_size=32,
          validation_split=0.2,
          callbacks=[early_stop, lr_reducer]
)


# 6.dumping the model and preprocessor
model.save(MODEL_FILE)
joblib.dump(tokenizer,TOKENIZER_FILE)


print(f"success! Model saved to {MODEL_FILE} and Tokenizer saved to {TOKENIZER_FILE}")