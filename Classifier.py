import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
import librosa

def job():
    df = pd.read_csv('train.csv')

    Classes = df.Class.unique().tolist()
    y = []
    X = []; yp = []; new_X = []
    for i in df.ID:
        new, rate = librosa.load('./train/Train/%d.wav'%i)
        mfccs = np.mean(librosa.feature.mfcc(y=new, sr=rate, n_mfcc=200).T, axis = 0)
        idx = Classes.index(df[df.ID == i]['Class'].tolist()[0])
        yp.append(idx)
        print i
        new_X.append(mfccs)
    return new_X, yp

t_beg = time.time()

NewX, y = job()

NewX_train, NewX_test, Newy_train, Newy_test = train_test_split(NewX, y, test_size=0.2, shuffle = True)
print 'successfully splitted'
t0 = time.time()
print 'time elapsed for reading: ', t0-t_beg

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(NewX_train, Newy_train)
t1 = time.time()
print 'time elapsed for fitting: ', t1-t0
print 'done fitting'

print 'fit to train new: ', clf.score(NewX_train, Newy_train)
print 'fit to test: ', clf.score(NewX_test, Newy_test)
