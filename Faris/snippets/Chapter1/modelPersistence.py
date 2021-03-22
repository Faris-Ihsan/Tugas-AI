#Menggunakan Pickle
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)


import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
print(y[0])



# In[]

#Menggunakan Joblib

#Buat Direktori temporary (sementara)
from tempfile import mkdtemp
savedir = mkdtemp()
import os
filename = os.path.join(savedir, 'test.joblib')

#Membuat objek yang akan di persist
import numpy as np
to_persist = [('a', [1, 2, 3]), ('b', np.arange(10))]

#simpan objek yang akan di persist ke filename
import joblib
joblib.dump(to_persist, filename) 

#load object dari file 
joblib.load(filename)