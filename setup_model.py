import os
import requests
import zipfile
import io
import sys
import shutil

# Link Model Asli dari GitHub Releases Anda
MODEL_URL = "https://github.com/ahnaf242/final-project-alpro-2/releases/download/v1.0/my_bert_model.zip"

# Nama folder yang diharapkan keluar dari ZIP
TARGET_FOLDER = "my_bert_model"

def setup():
    # 1. Cek apakah model sudah ada
    if os.path.exists(TARGET_FOLDER) and os.path.exists(os.path.join(TARGET_FOLDER, "config.json")):
        print(f"✅ Folder '{TARGET_FOLDER}' sudah ada dan lengkap! Tidak perlu download lagi.")
        return

    print("="*60)
    print("🚀 SENTIMEN AI - AUTO SETUP INSTALATION")
    print("="*60)
    print(f"📡 Sedang menghubungi GitHub Releases...")
    print(f"🔗 URL: {MODEL_URL}")
    print("⏳ Mohon tunggu, sedang mendownload model AI (~450 MB)...")
    print("   (Kecepatan tergantung koneksi internet Anda)")

    try:
        # Download Stream
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        
        print("📦 Sedang mengekstrak file ZIP...")
        
        # Ekstrak
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(".")
            
        # VERIFIKASI HASIL EKSTRAK
        if os.path.exists(TARGET_FOLDER):
            print("\n✅ SUKSES! Model berhasil terpasang.")
            print(f"📂 Lokasi: {os.path.abspath(TARGET_FOLDER)}")
            print("👉 Sekarang Anda siap menjalankan: python app.py")
        else:
            # Fallback jika user salah nge-zip (misal zip isinya flat files tanpa folder)
            print("\n⚠️ Peringatan: Struktur ZIP mungkin berbeda.")
            print("Mencoba merapikan file...")
            
            # Buat folder manual dan pindahkan file json/h5/txt ke sana
            os.makedirs(TARGET_FOLDER, exist_ok=True)
            for file in os.listdir("."):
                if file.endswith(".h5") or file.endswith(".json") or file.endswith(".txt"):
                    if file not in ["requirements.txt", "README.md"]: # Jangan pindahkan file project
                        shutil.move(file, os.path.join(TARGET_FOLDER, file))
            
            print("✅ Perbaikan selesai. Model berhasil terpasang.")

    except Exception as e:
        print(f"\n❌ GAGAL DOWNLOAD: {e}")
        print("💡 Solusi Alternatif:")
        print(f"1. Download manual dari: {MODEL_URL}")
        print(f"2. Ekstrak zip tersebut dan rename foldernya jadi '{TARGET_FOLDER}'")
        print("3. Taruh di sebelah file app.py")

if __name__ == "__main__":
    setup()