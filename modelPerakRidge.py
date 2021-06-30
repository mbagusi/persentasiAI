from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
from connect import data_perak as data
import datetime
import pandas as pd
import matplotlib.pyplot as  plt

besok = datetime.date.today() + datetime.timedelta(days=1)

date = data["date"] #(tipe data series)
date_predict = np.array([str(str(besok.year)+'-0'+str(besok.month)+'-'+str(besok.day))]) #"2021-06-30" (numpy array string)
price = data["price"]

def to_datetime(date):
    df = pd.DataFrame({'date': date})
    #merubah ke tipe data datetime
    df.date = pd.to_datetime(df.date)
    return df.date

def to_datetime_pred(date_predict):
    dfe = pd.DataFrame({'prediksi': date_predict})
    #merubah ke tipe data datetime
    dfe.prediksi = pd.to_datetime(dfe.prediksi)
    return dfe.prediksi

x = to_datetime(date).values.astype(float).reshape(-1, 1)
x_predict = to_datetime_pred(date_predict).values.astype(float).reshape(-1, 1)
y = price.values.reshape(-1, 1)

lin = LinearRegression()
lin.fit(x, y)

def coef_intercept(lin):
    coef = lin.coef_
    intercept = lin.intercept_
    return coef, intercept

def prediction(lin):
    lin_predict = lin.predict(x)
    lin_pred_future = lin.predict(x_predict)
    return lin_predict, lin_pred_future

coef, intercept = coef_intercept(lin)
lin_predict, lin_pred_future = prediction(lin)

plt.scatter(to_datetime(date), y, color='green')
plt.plot(to_datetime(date), lin_predict)
plt.plot(to_datetime_pred(date_predict), lin_pred_future)
plt.tick_params(labelrotation=30)
plt.ylabel("Dalam Rupiah")
plt.xlabel("Tanggal (jangka 14 hari)")
plt.legend(['Garis linear regression'])
plt.title("Grafik Linear Regression Prediksi Harga Perak")
plt.show()

print("Pediksi Harga Perak menggunakan metode Linear Regression")
print("Intercept = {}".format(intercept))
print("Coeffisien = {}".format(coef))
print("Prediksi Besoknya = {}".format(lin_pred_future))


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

rid = Ridge(alpha=0.01)
rid.fit(x, y)

def coef_intercept(rid):
    coef = rid.coef_
    intercept = rid.intercept_
    return coef, intercept

def prediction(rid):
    rid_predict = rid.predict(x)
    rid_pred_future = rid.predict(x_predict)
    return rid_predict, rid_pred_future

coef, intercept = coef_intercept(rid)
rid_predict, rid_pred_future = prediction(rid)

plt.scatter(to_datetime(date), y, color='green')
plt.plot(to_datetime(date), rid_predict)
plt.plot(to_datetime_pred(date_predict), rid_pred_future)
plt.tick_params(labelrotation=30)
plt.ylabel("Dalam Rupiah")
plt.xlabel("Tanggal (jangka 14 hari)")
plt.legend(['Garis Ridge regression'])
plt.title("Grafik Ridge Regression Prediksi Harga Perak")
plt.show()

print("Pediksi Harga Perak menggunakan metode Ridge Regression")
print("Intercept = {}".format(intercept))
print("Coeffisien = {}".format(coef))
print("Prediksi Besoknya = {}".format(rid_pred_future))
