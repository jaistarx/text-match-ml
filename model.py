
import numpy as np
import re
from numpy import asarray



import tensorflow_hub as hub


embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
def get_features(x):
    embeddings = embed(x)
    return asarray(embeddings)


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

import pandas as pd
data =pd.read_csv("train.csv" ,usecols=["description_x" ,"description_y"])
k= []

for x in range(len(data["description_x"])):
  k.append(test_similarity([data["description_x"][x]],[data["description_y"][x]]))


import pandas as pd
from numpy import asarray
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

def get_keras_model():
    """Define the model."""
    model = Sequential()
    model.add(Dense(128, input_shape=[x_train.shape[1]], activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model

categories=pd.read_csv("train.csv" ,usecols=["same_security"])
k=pd.DataFrame(k,columns=["k"])

x_train, x_test, y_train ,y_test =train_test_split(k, categories, shuffle=True)

y_train = asarray(y_train, dtype="float32")
y_test = asarray(y_test, dtype="float32")
model = get_keras_model()

model.fit(x_train, y_train, epochs=50, validation_split=0.2)

model.save("textmatch.h5")