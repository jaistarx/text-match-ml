
from flask import Flask, request,render_template

import numpy as np
import re
from numpy import asarray

import tensorflow_hub as hub


embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
def get_features(x):
    embeddings = embed(x)
    return asarray(embeddings)
app = Flask(__name__)

def cosines(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)


def test_similarity(text1, text2):
    vec1 = get_features(text1)[0]
    vec2 = get_features(text2)[0]
    return cosines(vec1, vec2)
from tensorflow.keras.models import load_model
import os
model=load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),"Textmatch.h5"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/index.html')
def indx():
    return render_template('index.html')
@app.route('/goals.html')
def goals():
    return render_template('goals.html')
@app.route('/academics.html')
def academics():
    return render_template('academics.html')
@app.route('/aboutme.html')
def aboutme():
    return render_template('aboutme.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    sim=test_similarity([int_features[0]], [int_features[1]])
    a = model.predict([float(sim)])
    if a[0][1] > 0.46:
        output='same'
    else:
        output='diff'
    return render_template('goals.html', prediction_text= output)

if __name__ == "__main__":
    app.run(debug=True)