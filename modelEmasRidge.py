from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from ambilData import data_emas as data
besok = datetime.date.today() + datetime.timedelta(days=1)

tanggal_besoknya = np.array([str(str(besok.year)+'-'+str(besok.month)+'-'+str(besok.day))]) #macam ni la misalnye(2021-07-09)
tanggal = data["date"]
harga = data["price"]

def merubah_ke_tipe_data_datetime(tanggal):
    tipe_data_dataframe = pd.DataFrame({'tanggal' : tanggal})
    tipe_data_datetime = pd.to_datetime(tipe_data_dataframe.tanggal)
    return tipe_data_datetime

def merubah_ke_tipe_data_datetime_besoknya(tanggal_besoknya):
    tipe_data_dataframe = pd.DataFrame({'tanggal_besoknya' : tanggal_besoknya})
    tipe_data_datetime = pd.to_datetime(tipe_data_dataframe.tanggal_besoknya)
    return tipe_data_datetime

uji_tanggal = merubah_ke_tipe_data_datetime(tanggal=tanggal)
uji_tanggal_besoknya = merubah_ke_tipe_data_datetime_besoknya(tanggal_besoknya=tanggal_besoknya)

#selanjutnya kita menginisiasi x train dan y train
x = uji_tanggal.values.astype(float).reshape(-1, 1)
y = harga.values.reshape(-1, 1)
x_predict = uji_tanggal_besoknya.values.astype(float).reshape(-1, 1)

#selanjutnya menginisasi fungsi machine learning, yaitu disini menggunakan linear regression
ridge = Ridge(alpha=0.01)
ridge.fit(x, y)

def coef_dan_intercept(ridge):
    coef = ridge.coef_
    intercept = ridge.intercept_
    return coef, intercept

def prediksi(ridge, x, x_predict):
    ridge_predict = ridge.predict(x)
    ridge_predict_besoknya = ridge.predict(x_predict)
    return ridge_predict, ridge_predict_besoknya

uji_klinis_coef, uji_klinis_intercept = coef_dan_intercept(ridge)
uji_klinis_ridge_predict, uji_klinis_ridge_predict_besoknya = prediksi(ridge, x, x_predict)

print("coef = " + str(uji_klinis_coef))
print("intecept = " + str(uji_klinis_intercept))
pred = uji_klinis_ridge_predict
print("prediksi besoknya = " + str(uji_klinis_ridge_predict_besoknya))

plt.scatter(uji_tanggal, harga, color="green")
plt.plot(uji_tanggal, uji_klinis_ridge_predict, color="red")
plt.plot(uji_tanggal_besoknya, uji_klinis_ridge_predict_besoknya)
plt.xlabel("Tanggal(interval 2minggu)")
plt.ylabel("Harga")
plt.title("prediksi harga emas menggunakan metode ridge regression")
plt.legend(["Garis linear"])
plt.tick_params(labelrotation=10)
plt.show()