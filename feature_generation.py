import pandas as pd
import numpy as np
from preprocessing import preprocess, preprocessBOW
sundanese = pd.read_csv('newdataset.csv')

fitur = sundanese.iloc[:, 1].values
labels = sundanese.iloc[:, 0].values

fitur_ekstraksi = preprocessBOW(fitur)
pd.DataFrame(fitur_ekstraksi).to_csv('feature_bow.csv',header=None,index=None,sep=',',quoting=1)
