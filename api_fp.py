from flask import Flask, url_for, render_template, redirect, request
from fp import predict_emotion

app = Flask(__name__)

@app.route('/<emo>/<sco>/', methods=['GET','POST'])
def emosi(teks):
    if request.method == 'POST':
        teks = request.form['']
        emo, sco = predict_emotion(teks)
        return redirect(url_for('', emo=emo, sco=sco[0]))
    return render_template('Alpro Front End.html')

