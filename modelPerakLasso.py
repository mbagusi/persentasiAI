# Medtode Lasso
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from connect import data_perak as data
import datetime
import pandas as pd
import matplotlib.pyplot as  plt

besok = datetime.date.today() + datetime.timedelta(days=1)

date = data["date"]
date_predict = np.array([str(str(besok.year)+'-0'+str(besok.month)+'-'+str(besok.day))])
price = data["price"]

date.head(14)

price.head(14)

def to_datetime(date):
    df = pd.DataFrame({'date': date})
    df.date = pd.to_datetime(df.date)
    return df.date

def to_datetime_pred(date_predict):
    dfe = pd.DataFrame({'prediksi': date_predict})
    dfe.prediksi = pd.to_datetime(dfe.prediksi)
    return dfe.prediksi

x = to_datetime(date).values.astype(float).reshape(-1, 1)
x_predict = to_datetime_pred(date_predict).values.astype(float).reshape(-1, 1)
y = price.values.reshape(-1, 1)


las = Lasso(alpha=0.01, tol=1, normalize=True)
las.fit(x, y)

def coef_intercept(las):
    coef = las.coef_
    intercept = las.intercept_
    return coef, intercept

def prediction(las):
    las_predict = las.predict(x)
    las_pred_future = las.predict(x_predict)
    return las_predict, las_pred_future

coef, intercept = coef_intercept(las)
las_predict, las_pred_future = prediction(las)

plt.scatter(to_datetime(date), y, color='green')
plt.plot(to_datetime(date), las_predict)
plt.plot(to_datetime_pred(date_predict), las_pred_future)
plt.tick_params(labelrotation=30)
plt.ylabel("Dalam Rupiah")
plt.xlabel("Tanggal (jangka 14 hari)")
plt.legend(['Garis Lasso'])
plt.title("Grafik Lasso Prediksi Harga Perak")
plt.show()

print("Pediksi Harga Perak menggunakan metode Lasso")
print("Intercept = {}".format(intercept))
print("Coeffisien = {}".format(coef))
print("Prediksi Besoknya = {}".format(lin_pred_future))

