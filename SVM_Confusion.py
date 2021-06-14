import pandas as pd
import numpy as np
from preprocessing import preprocess, extractTFIDF,extractBOW,extractNGram
sundanese = pd.read_csv('newdataset.csv')

fitur = sundanese.iloc[:, 1].values
labels = sundanese.iloc[:, 0].values

fitur_ekstraksi = extractNGram([1,2])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(fitur_ekstraksi, labels, train_size=0.8, random_state=0)

from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
klasifier = svm.SVC(kernel='linear', cache_size=1000, class_weight='balanced')

klasifierfit = OneVsOneClassifier(klasifier).fit(X_train, y_train)

hasil_prediksi = klasifierfit.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, hasil_prediksi))
print(classification_report(y_test, hasil_prediksi))
print(accuracy_score(y_test, hasil_prediksi))
