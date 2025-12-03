from flask import Flask, render_template, request
from fp import predict_emotion   # ambil fungsi dari backend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_label = None
    scores = None
    input_text = ""

    if request.method == "POST":
        # ambil teks dari form (name="teks")
        input_text = request.form.get("teks", "").strip()
        if input_text:
            label, score_array = predict_emotion(input_text)
            predicted_label = label
            # jadikan list biasa biar aman dipakai di template / JSON
            scores = [float(s) for s in score_array]

    # kirim data ke front end (index.html)
    return render_template(
        "index.html",
        input_text=input_text,
        predicted_label=predicted_label,
        scores=scores,
    )

if __name__ == "__main__":
    app.run(debug=True)
