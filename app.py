from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import os

app = Flask(__name__)

# =========================================================
# 1. LOAD RESOURCES (Model, Tokenizer, Kamus)
# =========================================================
print(">>> Loading Resources...")

# Menggunakan path absolut agar tidak bingung mencari file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

try:
    # Load Model
    model = load_model(get_path('model_emosi_lstm.h5'))
    
    # Load Tokenizer
    with open(get_path('tokenizer_emosi.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load Label Encoder
    with open(get_path('label_encoder.pickle'), 'rb') as handle:
        le = pickle.load(handle)

    # Load Kamus Singkatan
    df_kamus = pd.read_csv(get_path(os.path.join('data', 'kamus_singkatan.csv')), sep=';', header=None, names=['slang', 'formal'])
    slang_dict = dict(zip(df_kamus['slang'], df_kamus['formal']))
    
    print(">>> Resources Loaded Successfully!")

except Exception as e:
    print(f"[CRITICAL ERROR] Gagal memuat file resource: {e}")
    print("Pastikan file .h5 dan .pickle ada di folder yang sama dengan api_fp.py")

# =========================================================
# 2. FUNGSI CLEANING
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    return ' '.join(words)

# =========================================================
# 3. ROUTE & LOGIKA UTAMA
# =========================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    # Variabel default (kosong)
    prediksi_label = ""
    conf_score = ""
    original_text = ""

    if request.method == 'POST':
        # JURUS ANTI-CRASH:
        # Mencari input bernama 'tweet', kalau gak ada cari 'teks'
        # Kalau dua-duanya gak ada, isi dengan string kosong ""
        original_text = request.form.get('tweet') or request.form.get('teks') or ""
        
        # Hanya proses jika user benar-benar mengetik sesuatu
        if original_text.strip():
            try:
                # 1. Preprocessing
                clean = clean_text(original_text)
                
                # 2. Tokenizing
                seq = tokenizer.texts_to_sequences([clean])
                padded = pad_sequences(seq, maxlen=100) # Maxlen harus sama dengan training
                
                # 3. Prediksi
                hasil = model.predict(padded)
                label_index = np.argmax(hasil)
                
                # Ambil nama label dan confidence score
                prediksi_label = le.classes_[label_index]
                conf_score = f"{np.max(hasil)*100:.1f}%"
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                prediksi_label = "Error"

    # Render Template dengan variabel yang sesuai HTML
    return render_template('index.html', 
                           predicted_label=prediksi_label, 
                           confidence=conf_score,
                           input_text=original_text)

if __name__ == '__main__':
    app.run(debug=True)