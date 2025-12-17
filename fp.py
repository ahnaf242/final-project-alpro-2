# fp.py
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# --- PENGATURAN PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_resource_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)

# file hasil training
MODEL_PATH = get_resource_path("emotion_model.h5")
TOKENIZER_PATH = get_resource_path("tokenizer.pkl")
LABEL_ENCODER_PATH = get_resource_path("label_encoder.pkl")

# hyperparameter penting (harus konsisten train & prediksi)
MAX_WORDS = 10000
MAX_LEN = 100   # sama seperti punyamu tadi

# ======================================================
# 1. FUNGSI BACA DATA TXT 
# ======================================================
def load_txt(filename):
    filepath = get_resource_path(filename)
    
    texts = []
    labels = []
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File tidak ditemukan di: {filepath}. "
            "Pastikan file ada di folder yang sama dengan script ini."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(";")
                if len(parts) >= 2:
                    text = parts[0]
                    label = parts[1]
                    texts.append(text)
                    labels.append(label)
                    
    return pd.DataFrame({"text": texts, "label": labels})

# ======================================================
# 2. TRAINING + SAVE MODEL (.h5) + TOKENIZER + LABEL
# ======================================================
def train_and_save_model():
    print("Memuat data train/val/test...")
    train_df = load_txt("train.txt")
    val_df   = load_txt("val.txt")
    test_df  = load_txt("test.txt")

    print(f"Data Loaded: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

    # --- ENCODE LABEL ---
    label_encoder = LabelEncoder()
    train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
    val_df["label_enc"]   = label_encoder.transform(val_df["label"])
    test_df["label_enc"]  = label_encoder.transform(test_df["label"])

    # --- TOKENIZER ---
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text"])

    X_train_seq = tokenizer.texts_to_sequences(train_df["text"])
    X_val_seq   = tokenizer.texts_to_sequences(val_df["text"])
    X_test_seq  = tokenizer.texts_to_sequences(test_df["text"])

    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_val   = pad_sequences(X_val_seq, maxlen=MAX_LEN)
    X_test  = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    y_train = train_df["label_enc"].values
    y_val   = val_df["label_enc"].values
    y_test  = test_df["label_enc"].values

    num_classes = len(label_encoder.classes_)

    # --- DEFINISI MODEL  ---
    model = keras.Sequential([
        layers.Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        layers.SpatialDropout1D(0.3),
        layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.summary()

    print("Mulai Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=23,      # sama seperti sebelumnya
        batch_size=64,
        verbose=1
    )

    # --- EVALUASI ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    # --- SIMPAN MODEL + TOKENIZER + LABEL ENCODER ---
    print("\nMenyimpan model & artefak...")

    model.save(MODEL_PATH)
    print(f"Model disimpan di: {MODEL_PATH}")

    with open(TOKENIZER_PATH, "wb") as f_tok:
        pickle.dump(tokenizer, f_tok)
    print(f"Tokenizer disimpan di: {TOKENIZER_PATH}")

    with open(LABEL_ENCODER_PATH, "wb") as f_lbl:
        pickle.dump(label_encoder, f_lbl)
    print(f"Label encoder disimpan di: {LABEL_ENCODER_PATH}")

    print("\nTraining & penyimpanan selesai.")

# ======================================================
# 3. LOAD MODEL UNTUK PREDIKSI (DIPAKAI API)
# ======================================================

# variabel global (di-load sekali saja)
_model = None
_tokenizer = None
_label_encoder = None

def load_artifacts():
    """Load model, tokenizer, dan label encoder dari file .h5/.pkl (sekali saja)."""
    global _model, _tokenizer, _label_encoder

    if _model is None or _tokenizer is None or _label_encoder is None:
        print("Load model & artefak dari file...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model .h5 tidak ditemukan di {MODEL_PATH}. "
                "Jalankan dulu: python fp.py untuk training & save model."
            )

        _model = keras.models.load_model(MODEL_PATH)

        with open(TOKENIZER_PATH, "rb") as f_tok:
            _tokenizer = pickle.load(f_tok)

        with open(LABEL_ENCODER_PATH, "rb") as f_lbl:
            _label_encoder = pickle.load(f_lbl)

    return _model, _tokenizer, _label_encoder

def predict_emotion(text: str):
    """
    Fungsi yang dipanggil dari Flask (api_fp.py).
    TIDAK melakukan training, hanya load model + prediksi.
    """
    model, tokenizer, label_encoder = load_artifacts()

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded, verbose=0)
    label_idx = np.argmax(pred)
    label = label_encoder.inverse_transform([label_idx])[0]
    return label, pred[0]

# ======================================================
# 4. MODE TRAINING (JALANKAN MANUAL SEKALI)
# ======================================================
if __name__ == "__main__":
    # Kalo kamu jalankan: python fp.py
    # dia cuma training & simpan model, bukan jalan sebagai API.
    train_and_save_model()

    # opsional: tes 1 contoh
    print("\nContoh prediksi setelah training:")
    print(predict_emotion("Saya sangat senang hari ini"))
