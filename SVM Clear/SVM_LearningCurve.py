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
from sklearn.model_selection import KFold, learning_curve, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

sundanese = pd.read_csv('newdataset.csv')

fitur = sundanese.iloc[:,1].values
labels = sundanese.iloc[:,0].values

factory = StemmerFactory()
stemmer = factory.create_stemmer()

fitur_ekstraksi0 = []
for cuitan in range(0, len(fitur)):
    tmp = str(fitur[cuitan]).lower()
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

stopsunda1 = open('stopwordv1.txt', 'r')
stopsunda2 = stopsunda1.read()
stopsunda = word_tokenize(stopsunda2)

def swr(a, b):
    filtered_sentence = []
    for w in a:
        if w not in b:
            filtered_sentence.append(w)
    return filtered_sentence

def stopw(datanext):
    fitur_ekstraksiku = []
    for cuitan in range(0, len(datanext)):
        tmp = swr(datanext[cuitan], stopsunda)
        fitur_ekstraksiku.append(tmp)
    return fitur_ekstraksiku

fitur_ekstraksi = stopw(fitur_ekstraksi5)

def identity_tokenizer(text):
    return text

from sklearn.feature_extraction.text import TfidfVectorizer

vektor_kata = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
fitur_ekstraksi = vektor_kata.fit_transform(fitur_ekstraksi).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(fitur_ekstraksi, labels, train_size=0.8, random_state=0)

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
klasifier = svm.SVC(kernel='linear', cache_size=1000, class_weight='balanced')

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = "Learning Curves (SVM)"
estimator = OneVsOneClassifier(klasifier).fit(X_train, y_train)
plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
