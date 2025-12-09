import os
import sys

# ==========================================
# 0. KONFIGURASI SISTEM
# ==========================================
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import tensorflow as tf

try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except ImportError:
    print("❌ Error: Library belum lengkap. Pip install transformers tf-keras")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. SETUP DYNAMIC PATH (MAGIC CODE)
# ==========================================
# Ini adalah "JANGKAR". Dia akan mendeteksi lokasi app.py di laptop manapun.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    # Menggabungkan folder tempat app.py berada dengan nama file tujuan
    return os.path.join(BASE_DIR, filename)

print(f">>> [INFO] Aplikasi berjalan di folder: {BASE_DIR}")

# Global Variables
model = None
tokenizer = None
slang_dict = {}
LABELS = ['anger', 'fear', 'happy', 'love', 'sadness']

try:
    # --- LOAD MODEL (PORTABLE) ---
    # Kita cari folder 'my_bert_model' tepat di sebelah app.py
    MODEL_DIR = get_path('my_bert_model')
    
    print(f">>> [INIT] Mencari model di: {MODEL_DIR}")
    
    if os.path.exists(MODEL_DIR):
        # Cek apakah isinya lengkap (mencegah folder kosong)
        if not os.path.exists(os.path.join(MODEL_DIR, 'config.json')):
             raise FileNotFoundError("Folder model ketemu, tapi 'config.json' hilang!")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("✅ Model & Tokenizer BERHASIL dimuat!")
    else:
        raise FileNotFoundError(f"Folder 'my_bert_model' TIDAK DITEMUKAN di {BASE_DIR}")

    # --- LOAD KAMUS ---
    # Asumsi: folder 'data' ada di sebelah app.py
    KAMUS_DIR = get_path(os.path.join('data', 'kamus_singkatan.csv'))
    
    if os.path.exists(KAMUS_DIR):
        df_kamus = pd.read_csv(KAMUS_DIR, sep=';', header=None, names=['slang', 'formal'])
        slang_dict = dict(zip(df_kamus['slang'], df_kamus['formal']))
        print("✅ Kamus singkatan dimuat.")
    else:
        print(f"⚠️ Kamus tidak ditemukan di {KAMUS_DIR}. Lanjut tanpa kamus.")

except Exception as e:
    print("\n" + "!"*50)
    print(f"❌ [CRITICAL ERROR] {e}")
    print("!"*50 + "\n")

# ==========================================
# 2. PREPROCESSING BERT
# ==========================================
emoticon_patterns = {
    re.compile(r'(:\)|:-\)|\(:)'): ' emot_senyum ',
    re.compile(r'(:d|:D|😂|🤣)'): ' emot_tertawa ',
    re.compile(r'(:\(|:-\()'): ' emot_sedih ',
    re.compile(r'(:\'|T_T|😭|😢)'): ' emot_menangis ',
    re.compile(r'(-_-)'): ' emot_datar ',
    re.compile(r'(<3|❤|💕)'): ' emot_cinta ',
    re.compile(r'(😡|😠)'): ' emot_marah '
}

def clean_text_bert(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    for pattern, replacement in emoticon_patterns.items():
        text = pattern.sub(replacement, text)
    text = re.sub(r'@[\w]+', '', text) 
    text = re.sub(r'https?://\S+', '', text)
    words = text.split()
    if slang_dict:
        words = [slang_dict.get(w, w) for w in words]
    text = ' '.join(words)
    text = re.sub(r'[^a-z0-9!?.,\s]', '', text)
    return " ".join(text.split())

# ==========================================
# 3. ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/app')
def detector():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        if not model or not tokenizer:
            return jsonify({'status': 'error', 'message': 'Model gagal dimuat saat startup.'}), 500

        data = request.get_json()
        text = data.get('tweet', '')
        
        if not text: return jsonify({'status': 'error', 'message': 'Tweet kosong.'}), 400

        # Pipeline
        clean_txt = clean_text_bert(text)
        inputs = tokenizer(clean_txt, return_tensors="tf", truncation=True, padding=True, max_length=80)
        outputs = model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
        
        label_index = np.argmax(probs)
        prediksi_label = LABELS[label_index]
        confidence = float(probs[label_index] * 100)

        return jsonify({
            'status': 'success',
            'data': {
                'prediction': prediksi_label,
                'confidence': f"{confidence:.1f}%"
            }
        })

    except Exception as e:
        print(f"[API ERROR] {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print(">>> Server Flask Siap!")
    app.run(debug=True, port=5000)