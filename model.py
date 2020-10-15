
import pandas as pd
from numpy import asarray
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
data =pd.read_csv("train.csv" ,usecols=["description_x" ,"description_y"])
data["new"]=data["description_x"]+data["description_y"]
categories=pd.read_csv("train.csv" ,usecols=["same_security"])

def get_keras_model():
    """Define the model."""
    model = Sequential()
    model.add(Dense(128, input_shape=[512 ,], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64 ,activation='relu' ,kernel_regularizer=tf.keras.regularizers.L1(0.01),
                    activity_regularizer=tf.keras.regularizers.L2(0.01)))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    return model


x_train, x_test, y_train ,y_test =train_test_split(data["new"], categories, shuffle=True)

import tensorflow_hub as hub


embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

def get_embeddings(x):
    embeddings = embed(x)
    return asarray(embeddings)


train_encodings = get_embeddings(x_train.to_list())
test_encodings = get_embeddings(x_test.tolist())

y_train = asarray(y_train, dtype="float32")
y_test = asarray(y_test, dtype="float32")

model = get_keras_model()
print(train_encodings.shape)
model.fit(train_encodings, y_train, epochs=50, validation_split=0.2)

model.save("TextMatch.h5")

score, acc = model.evaluate(test_encodings, y_test)
