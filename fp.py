import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# --- BAGIAN KRUSIAL: PENGATURAN PATH DINAMIS ---
# Mendapatkan lokasi folder tempat file script ini (fp.py) berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_resource_path(filename):
    """Fungsi helper untuk menggabungkan folder script dengan nama file data."""
    return os.path.join(BASE_DIR, filename)

def load_txt(filename):
    # Menggunakan path dinamis
    filepath = get_resource_path(filename)
    
    texts = []
    labels = []
    
    # Cek apakah file ada sebelum dibuka untuk menghindari crash
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan di: {filepath}. Pastikan file ada di folder yang sama dengan script ini.")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Menggunakan try-except untuk menangani baris yang formatnya rusak
                try:
                    # split(";", 1) membatasi pemisahan hanya pada semicolon pertama/terakhir 
                    # (tergantung struktur datamu, ini asumsi aman text;label)
                    parts = line.split(";")
                    if len(parts) >= 2:
                        text = parts[0]
                        label = parts[1]
                        texts.append(text)
                        labels.append(label)
                except ValueError:
                    continue # Skip baris yang error
                    
    return pd.DataFrame({"text": texts, "label": labels})

# --- LOAD DATA MENGGUNAKAN NAMA FILE SAJA ---
# Tidak perlu lagi C:/Users/Acer/... cukup nama filenya.
print("Memuat data...")
train_df = load_txt("train.txt")
val_df   = load_txt("val.txt")
test_df  = load_txt("test.txt")

print(f"Data Loaded: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

# --- PREPROCESSING ---
label_encoder = LabelEncoder()
train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
val_df["label_enc"]   = label_encoder.transform(val_df["label"])
test_df["label_enc"]  = label_encoder.transform(test_df["label"])

max_words = 10000   # vocab size
max_len = 100       # max words per sentence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

X_train = tokenizer.texts_to_sequences(train_df["text"])
X_val   = tokenizer.texts_to_sequences(val_df["text"])
X_test  = tokenizer.texts_to_sequences(test_df["text"])

X_train = pad_sequences(X_train, maxlen=max_len)
X_val   = pad_sequences(X_val, maxlen=max_len)
X_test  = pad_sequences(X_test, maxlen=max_len)

y_train = train_df["label_enc"].values
y_val   = val_df["label_enc"].values
y_test  = test_df["label_enc"].values

# --- MODEL DEFINITION ---
# Catatan: recurrent_dropout membuat training lambat jika tidak pakai GPU khusus.
# Jika terasa sangat lambat, set recurrent_dropout=0
model = keras.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    layers.SpatialDropout1D(0.3),
    layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3), 
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.summary()

# --- TRAINING ---
print("Mulai Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=23,
    batch_size=64,
    verbose=1
)

# --- EVALUATION ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    label_idx = np.argmax(pred)
    label = label_encoder.inverse_transform([label_idx])[0]
    return label, pred[0]

# Contoh penggunaan
print(predict_emotion("Saya sangat senang hari ini"))