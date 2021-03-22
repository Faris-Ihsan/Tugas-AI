from sklearn import datasets #ambil fungsi datasets dari library sklearn
iris = datasets.load_iris() #load data dari dataset iris
digits = datasets.load_digits() #load data dari dataset digits
print(digits.target[0]) #Cetak data digits dengan key target[0] 
print(digits.data[0]) #Cetak data digits dengan key data[0] 
print(digits.images[0]) #Cetak data digits dengan key images[0]


# In[]
from sklearn import svm  #ambil fungsi svm dari sklearn
clf = svm.SVC(gamma=0.001, C=100.) #parameter model
clf.fit(digits.data[:-1], digits.target[:-1]) #masukan x dan y
clf.predict(digits.data[-1:]) #lakukan prediksi

