import numpy as np
from flask import Flask, request, jsonify, render_template
from numpy import asarray
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import os

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

def get_embeddings(x):
    embeddings = embed(x)
    return asarray(embeddings)

app = Flask(__name__)
model=load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),"TextMatch.h5"))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    k = int_features[0] + ' ' + int_features[1]
    k = get_embeddings([k])
    a = model.predict(k)
    if a[0][1] > 0.40:
        output ="same"+str(a[0][1])
    else:
        output="diff"+str(a[0][1])


    return render_template('index.html', prediction_text= output)

if __name__ == "__main__":
    app.run(debug=True)