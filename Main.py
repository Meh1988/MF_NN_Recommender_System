import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


# Load data
ratings_data = pd.read_csv('ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies_data = pd.read_csv('movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'])

# Merge data
data = pd.merge(ratings_data, movies_data, on='MovieID')

# Preprocess data
num_users = data['UserID'].nunique()
num_movies = data['MovieID'].nunique()

user_mapping = {user_id: idx for idx, user_id in enumerate(data['UserID'].unique())}
movie_mapping = {movie_id: idx for idx, movie_id in enumerate(data['MovieID'].unique())}

data['UserID'] = data['UserID'].map(user_mapping)
data['MovieID'] = data['MovieID'].map(movie_mapping)

X = data[['UserID', 'MovieID']].values
y = data['Rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(num_users, num_movies, embedding_dim=32):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
    movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim)(movie_input)
    
    user_flat = Flatten()(user_embedding)
    movie_flat = Flatten()(movie_embedding)
    
    concat = Concatenate()([user_flat, movie_flat])
    
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    
    output = Dense(1, activation='linear')(dense2)
    
    model = Model(inputs=[user_input, movie_input], outputs=output)
    return model

model = build_model(num_users, num_movies)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()



model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1)


# Model Evaluation
y_pred = model.predict([X_test[:, 0], X_test[:, 1]])

# Mean Squared Error
mse = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print('Mean Squared Error:', mse)

# Precision, Recall, F1, and nDCG
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ndcg_score

# Convert predicted ratings to binary values (e.g., Liked or Not Liked)
y_pred_binary = (y_pred >= 3.5).astype(int)
y_test_binary = (y_test >= 3.5).astype(int)

precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)



# Print evaluation metrics
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
