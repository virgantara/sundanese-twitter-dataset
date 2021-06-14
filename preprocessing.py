import re
from string import digits
import string
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def swr(a, b, stemmer):
    filtered_sentence = []
    for w in a:
        if w not in b:
            t = stemmer.stem(w)
            filtered_sentence.append(t)
    return filtered_sentence

def preprocess(data):

    stopsunda = open('stopwordv1.txt', 'r')
    stopsunda = stopsunda.read()
    stopsunda = word_tokenize(stopsunda)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    fitur_ekstraksi = []
    for cuitan in range(0, len(data)):
        tmp = str(data[cuitan]).lower()
        #
        tmp = tmp.translate(str.maketrans(' ', ' ', digits))  # membuang karakter angka
        tmp = tmp.translate(str.maketrans(' ', ' ', string.punctuation))  # membuang karakter
        tmp = re.sub(r'\W', ' ', str(tmp))  # membuang karakter khusus selain angka dan huruf
        tmp = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(tmp))  # membuang kata yang hanya satu huruf
        tmp = re.sub(r'\^[a-zA-Z]\s+', ' ',
                     str(tmp))  # membuang kata yang hanya satu huruf dari awal
        tmp = re.sub(r'\s+', ' ', str(tmp))  # mengganti spasi ganda dengan spasi tunggal

        tmp = word_tokenize(str(tmp)) # for TFIDF
        tmp = swr(tmp, stopsunda, stemmer) # for TFIDF
        print(tmp)


        fitur_ekstraksi.append(tmp)


    return fitur_ekstraksi

def preprocessBOW(data):

    stopsunda = open('stopwordv1.txt', 'r')
    stopsunda = stopsunda.read()
    stopsunda = word_tokenize(stopsunda)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    fitur_ekstraksi = []
    for cuitan in range(0, len(data)):
        tmp = str(data[cuitan]).lower()
        #
        tmp = tmp.translate(str.maketrans(' ', ' ', digits))  # membuang karakter angka
        tmp = tmp.translate(str.maketrans(' ', ' ', string.punctuation))  # membuang karakter
        tmp = re.sub(r'\W', ' ', str(tmp))  # membuang karakter khusus selain angka dan huruf
        tmp = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(tmp))  # membuang kata yang hanya satu huruf
        tmp = re.sub(r'\^[a-zA-Z]\s+', ' ',
                     str(tmp))  # membuang kata yang hanya satu huruf dari awal
        tmp = re.sub(r'\s+', ' ', str(tmp))  # mengganti spasi ganda dengan spasi tunggal
        # tmp = stemmer.stem(tmp) # for BOW
        tmp = word_tokenize(str(tmp))  # for TFIDF
        tmp = swr(tmp, stopsunda, stemmer)  # for TFIDF

        fitur_ekstraksi.append(tmp)


    return fitur_ekstraksi

def identity_tokenizer(text):
    return text

def extractTFIDF():
    features = pd.read_csv('feature_tfidf.csv', sep=',', header=None, quoting=1)  #

    fitur_ekstraksi = []
    df = pd.DataFrame(features)
    for row in df.values:
        tmp = [x for x in row if str(x) != 'nan']
        fitur_ekstraksi.append(tmp)

    vektor_kata = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    fitur_ekstraksi = vektor_kata.fit_transform(fitur_ekstraksi).toarray()

    return fitur_ekstraksi

def extractBOW():
    features = pd.read_csv('feature_bow.csv',header=None)  #
    sentences = features.values.reshape(-1)

    vec = CountVectorizer()
    data = vec.fit_transform(sentences)

    return data

def extractNGram(params):
    features = pd.read_csv('feature_bow.csv', header=None)  #
    sentences = features.values.reshape(-1)

    vec = CountVectorizer(ngram_range=params)
    data = vec.fit_transform(sentences)
    return data