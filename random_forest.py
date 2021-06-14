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
from preprocessing import preprocess, extractTFIDF, extractBOW, extractNGram

sundanese = pd.read_csv('newdataset.csv')

fitur = sundanese.iloc[:,1].values
labels = sundanese.iloc[:,0].values

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# fitur_ekstraksi = preprocess(fitur)
fitur_ekstraksi = extractNGram([1,2])
# print(fitur_ekstraksi)
#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(fitur_ekstraksi, labels, train_size=0.8, random_state=0)

from sklearn.ensemble import RandomForestClassifier

klasifier = RandomForestClassifier(n_estimators=200, random_state=0)
klasifier.fit(X_train, y_train)

hasil_prediksi = klasifier.predict(X_test)
# print(*hasil_prediksi[:5])
# print('\n')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,hasil_prediksi))
print(classification_report(y_test,hasil_prediksi))
print(accuracy_score(y_test, hasil_prediksi))

# from sklearn.metrics import plot_confusion_matrix
# titles_options = [("Confusion matrix", None)]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(klasifier, X_test, y_test,
#                                  display_labels=labels,
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize,
#                                  values_format='g')
#     disp.ax_.set_title(title)
#
# plt.show()