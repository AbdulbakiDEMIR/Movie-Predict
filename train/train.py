import kagglehub
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping

# === 1. VERİ YÜKLEME ===
path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
movies = pd.read_csv(path+"/movies.csv")
ratings = pd.read_csv(path+"/ratings.csv")

# === 2. VERİ HAZIRLAMA ===
data = ratings.merge(movies, on="movieId")
data["genres"] = data["genres"].apply(lambda x: x.split('|'))

# Kullanıcı ve film sayısı çok az olanları filtrele
valid_users = data["userId"].value_counts()
valid_movies = data["movieId"].value_counts()
data = data[
    data["userId"].isin(valid_users[valid_users > 10].index) &
    data["movieId"].isin(valid_movies[valid_movies > 10].index)
]

# Encode user/movie
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()
data["user_enc"] = user_encoder.fit_transform(data["userId"])
data["movie_enc"] = movie_encoder.fit_transform(data["movieId"])

# Genre vectorizer
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(data["genres"])

# Eğitim girdileri
user_ids = data["user_enc"].values
movie_ids = data["movie_enc"].values
ratings_val = data["rating"].values

# === 3. MODEL MİMARİSİ ===
num_users = data["user_enc"].nunique()
num_movies = data["movie_enc"].nunique()
num_genres = genre_matrix.shape[1]

user_input = Input(shape=(1,), name="user_input")
movie_input = Input(shape=(1,), name="movie_input")
genre_input = Input(shape=(num_genres,), name="genre_input")

user_vec = Flatten()(Embedding(num_users, 64)(user_input))
movie_vec = Flatten()(Embedding(num_movies, 64)(movie_input))

x = Concatenate()([user_vec, movie_vec, genre_input])
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
output = Dense(1, name="rating_output")(x)

model = Model(inputs=[user_input, movie_input, genre_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0003), loss=Huber())

# === 4. ERKEN DURDURMA ===
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# === 5. MODEL EĞİTİMİ ===
model.fit(
    [user_ids, movie_ids, genre_matrix],
    ratings_val,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop]
)

# === 6. MODEL VE ENCODER'LARI KAYDET ===
model.save("movie_predict_model.h5")
print("✅ Model kaydedildi: movie_predict_model.h5")

with open("movie_encoder.pkl", "wb") as f:
    pickle.dump(movie_encoder, f)
with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
print("✅ movie_encoder.pkl ve mlb.pkl kaydedildi.")

