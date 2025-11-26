import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def load_txt(filepath):
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # avoid empty lines
                text, label = line.split(";")
                texts.append(text)
                labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})

train_df = load_txt("D:/SEM 3/Alpro 2/train.txt")
val_df   = load_txt("D:/SEM 3/Alpro 2/val.txt")
test_df  = load_txt("D:/SEM 3/Alpro 2/test.txt")

label_encoder = LabelEncoder()
train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
val_df["label_enc"]   = label_encoder.transform(val_df["label"])
test_df["label_enc"]  = label_encoder.transform(test_df["label"])

max_words = 10000   # vocab size
max_len = 100       # max words per sentence

tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

X_train = tokenizer.texts_to_sequences(train_df["text"])
X_val   = tokenizer.texts_to_sequences(val_df["text"])
X_test  = tokenizer.texts_to_sequences(test_df["text"])

X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_val   = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_len)
X_test  = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)

y_train = train_df["label_enc"].values
y_val   = val_df["label_enc"].values
y_test  = test_df["label_enc"].values

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

model.build(input_shape=(None, max_len))
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=23,
    batch_size=64,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)

def predict_emotion(text):
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    
    # Predict using your trained model
    pred = model.predict(padded)
    
    # Get label
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0], pred[0]