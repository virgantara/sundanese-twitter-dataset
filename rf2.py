import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import Sastrawi.Stemmer
import string
from nltk.tokenize import word_tokenize
from string import digits
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

sundanese = pd.read_csv('newone.csv')
sundatest = pd.read_csv('testdataset.csv')

fitur = sundanese.iloc[:,1].values
labels = sundanese.iloc[:,0].values
fitur2 = sundatest.iloc[:,1].values
labels2 = sundatest.iloc[:,0].values

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(data):
    fitur_ekstraksi0 = []
    for cuitan in range(0, len(data)):
        tmp = str(data[cuitan]).lower()
        fitur_ekstraksi0.append(tmp)

    fitur_ekstraksi1 = []
    for cuitan in range(0, len(fitur_ekstraksi0)):
        tmp = fitur_ekstraksi0[cuitan].translate(str.maketrans(' ', ' ', digits))  # membuang karakter angka
        fitur_ekstraksi1.append(tmp)

    fitur_ekstraksi2 = []
    for cuitan in range(0, len(fitur_ekstraksi1)):
        tmp = fitur_ekstraksi1[cuitan].translate(str.maketrans(' ', ' ', string.punctuation))  # membuang karakter
        fitur_ekstraksi2.append(tmp)

    fitur_ekstraksi3 = []
    # fitur_ekstraksi = []
    for cuitan in range(0, len(fitur_ekstraksi2)):
        tmp = re.sub(r'\W', ' ',str(fitur_ekstraksi2[cuitan])) # membuang karakter khusus selain angka dan huruf
        tmp = re.sub(r'\s+[a-zA-Z]\s+', ' ',str(fitur_ekstraksi2[cuitan])) # membuang kata yang hanya satu huruf
        tmp = re.sub(r'\^[a-zA-Z]\s+', ' ',str(fitur_ekstraksi2[cuitan])) # membuang kata yang hanya satu huruf dari awal
        tmp = re.sub(r'\s+', ' ',str(fitur_ekstraksi2[cuitan])) # mengganti spasi ganda dengan spasi tunggal
        fitur_ekstraksi3.append(tmp)
        # fitur_ekstraksi.append(tmp)

    # fitur_ekstraksi4 = []
    # for cuitan in range(0, len(fitur_ekstraksi3)):
    #     tmp = stemmer.stem(str(fitur_ekstraksi3[cuitan]))
    #     fitur_ekstraksi4.append(tmp)

    fitur_ekstraksi5 = []
    for cuitan in range(0, len(fitur_ekstraksi3)):
        tmp = word_tokenize(str(fitur_ekstraksi3[cuitan]))
        fitur_ekstraksi5.append(tmp)

    return fitur_ekstraksi5

stopsunda1 = open('stopwordv1.txt', 'r')
stopsunda2 = stopsunda1.read()
stopsunda = word_tokenize(stopsunda2)


def swr(a, b):
    filtered_sentence = []
    for w in a:
        if w not in b:
            filtered_sentence.append(w)
    return filtered_sentence

callbackvalue = preprocessing(fitur)
callbackvalue1 = preprocessing(fitur2)

def stopw(datanext):
    fitur_ekstraksiku = []
    for cuitan in range(0, len(datanext)):
        tmp = swr(datanext[cuitan], stopsunda)
        fitur_ekstraksiku.append(tmp)
    return fitur_ekstraksiku

fitur_ekstraksinext = stopw(callbackvalue)
fitur_ekstraksinext2 = stopw(callbackvalue1)
print(fitur_ekstraksinext2[11])
print('\n')

# for cuitan in range(0, len(fitur_ekstraksinext2)):
#     tmp = fitur_ekstraksinext2[cuitan]
#     fitur_ekstraksinext.append(tmp)

def identity_tokenizer(text):
    return text

from sklearn.feature_extraction.text import TfidfVectorizer

vektor_kata = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
fitur_ekstraksisiap = vektor_kata.fit_transform(fitur_ekstraksinext).toarray()

fitur_ekstraksi = []
for cuitan in range(0, len(callbackvalue)):
    tmp = fitur_ekstraksisiap[cuitan]
    fitur_ekstraksi.append(tmp)

fitur_ekstraksitest = []
for cuitan in range(len(callbackvalue), len(fitur_ekstraksisiap)):
    tmp = fitur_ekstraksisiap[cuitan]
    fitur_ekstraksitest.append(tmp)

# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(fitur_ekstraksi, labels, train_size=0.99, random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# klasifier = RandomForestClassifier(n_estimators=200, random_state=0)
# klasifier.fit(fitur_ekstraksi, labels)
scores = []
best_svr = RandomForestClassifier(n_estimators=200, random_state=0)
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(fitur_ekstraksisiap):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = fitur_ekstraksisiap[train_index], fitur_ekstraksisiap[test_index], labels[train_index], labels[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))
print(scores)
print(np.mean(scores))
# hasil_prediksi = klasifier.predict(fitur_ekstraksitest)
# print(hasil_prediksi)
# print('\n')
#
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# print(confusion_matrix(labels2,hasil_prediksi))
# print(classification_report(labels2,hasil_prediksi))
# print(accuracy_score(labels2, hasil_prediksi))