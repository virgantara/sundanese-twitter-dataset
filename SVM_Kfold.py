import numpy as np
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from string import digits
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import KFold, cross_val_predict

sundanese = pd.read_csv('newone.csv')

fitur = sundanese.iloc[:, 1].values
labels = sundanese.iloc[:, 0].values

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
    for cuitan in range(0, len(fitur_ekstraksi2)):
        tmp = re.sub(r'\W', ' ',str(fitur_ekstraksi2[cuitan])) # membuang karakter khusus selain angka dan huruf
        tmp = re.sub(r'\s+[a-zA-Z]\s+', ' ',str(fitur_ekstraksi2[cuitan])) # membuang kata yang hanya satu huruf
        tmp = re.sub(r'\^[a-zA-Z]\s+', ' ',str(fitur_ekstraksi2[cuitan])) # membuang kata yang hanya satu huruf dari awal
        tmp = re.sub(r'\s+', ' ',str(fitur_ekstraksi2[cuitan])) # mengganti spasi ganda dengan spasi tunggal
        fitur_ekstraksi3.append(tmp)

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

def stopw(datanext):
    fitur_ekstraksiku = []
    for cuitan in range(0, len(datanext)):
        tmp = swr(datanext[cuitan], stopsunda)
        fitur_ekstraksiku.append(tmp)
    return fitur_ekstraksiku

fitur_ekstraksinext = stopw(callbackvalue)

def identity_tokenizer(text):
    return text

from sklearn.feature_extraction.text import TfidfVectorizer

vektor_kata = TfidfVectorizer(max_features=5000, tokenizer=identity_tokenizer, lowercase=False)
X = vektor_kata.fit_transform(fitur_ekstraksinext).toarray()

from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
scores = []
best_svr = svm.SVC(kernel='linear', cache_size=1000, class_weight='balanced')
kunyuk = OneVsOneClassifier(best_svr)

cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], labels[train_index], labels[test_index]
    kunyuk.fit(X_train, y_train)
    scores.append(kunyuk.score(X_test, y_test))
print(scores)
print(np.mean(scores))