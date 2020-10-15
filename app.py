
from flask import Flask, request,render_template

import numpy as np
import re
from numpy import asarray



import tensorflow_hub as hub


embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
def get_features(x):
    embeddings = embed(x)
    return asarray(embeddings)

def process_text(text):
    text = text.encode('ascii', errors='ignore').decode()
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'#+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
    # text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"won't", "will not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


data= []
import io
with io.open('listfile.txt','r',encoding='utf-8') as f1:
    files =f1.readlines()
    for line in files:
        current_place=line[:-1]
        data.append(current_place)

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
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    sim=test_similarity([int_features[0]], [int_features[1]])
    if sim>0.5:
        output='same'
    else:
        output='diff'
    return render_template('index.html', prediction_text= output)

if __name__ == "__main__":
    app.run(debug=True)