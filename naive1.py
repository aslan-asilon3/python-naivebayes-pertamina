import pandas as pd
import numpy as np

# input data
Cryotherapy=pd.read_csv("Cryotherapy.csv")
# Menampilkan data
Cryotherapy.head()

# menampilkan informasi data
Cryotherapy.info()

# Mengecek apakah ada deret yang kosong
Cryotherapy.empty

# Melihat ukuran dari data
Cryotherapy.size

# Variabel independen
x = Cryotherapy.drop(["Result_of_Treatment"], axis = 1)
x.head()

# Variabel dependen
y = Cryotherapy["Result_of_Treatment"]
y.head()

# Import train_test_split function
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive bayes
modelnb = GaussianNB()
# Memasukkan data training pada fungsi klasifikasi naive bayes
nbtrain = modelnb.fit(x_train, y_train)
nbtrain.class_count_

# Menentukan hasil prediksi dari x_test
y_pred = nbtrain.predict(x_test)
y_pred