import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Leyendo el dataset y convirtiendo la columna de fecha a formato de fecha y como indice
series = pd.read_csv('NVDA2.csv',usecols=['Date','Close'])
series['Date'] = pd.to_datetime(series['Date'])
series = series.set_index('Date')

#Se plotea el dataset
plt.plot(series,label='Entrenamiento')
plt.title('Precio de acciones de Apple(USD)', fontsize = 20)
plt.xlabel('Fecha',fontsize = 15)
plt.ylabel('Precio', fontsize = 15)
plt.xlim(series.index.min(), series.index.max())
plt.xticks(rotation=30)
plt.tight_layout()
#Se define el numero de periodos que se van a predecir
periodos = 2
#Se crea un nuevo dataset en el que nuestras etiquetas seran tiempo que estaran numeradas de 0 hasta la longitud que tengan los datos, de esta manera sera mas facil predecir y entrenar al modelo
df = series.copy()
df['Time'] = np.arange(len(series.index))
#print(df.head())
X = df.loc[:,['Time']]
Y = df.loc[:,'Close']
#separamos el modelo en datos de prueba y datos de entrenamiento
Xprueba = X[-periodos:len(X)]
Yprueba = Y[-periodos:len(Y)]
Xentrenamiento = X[0:-periodos]
Yentrenamiento = Y[0:-periodos]

plt.plot(Yprueba,color='red',label='Prueba')
def predRegresionLineal(Xprueba,Yprueba,Xentrenamiento,Yentrenamiento):
    modelo = LinearRegression()
    modelo.fit(Xentrenamiento,Yentrenamiento)
    ypred = modelo.predict(Xprueba)
    predicciones = pd.Series(ypred, index=Xprueba.index)
    plt.plot(predicciones,color='green',label='Predicciones regresion lineal')
    mse = mean_squared_error(Yprueba.values,ypred)
    rmse = np.sqrt(mse)

def predARMA(Xprueba,Yprueba,Xentrenamiento,Yentrenamiento):
    modelo = SARIMAX(Yentrenamiento,order=(1,0,1))
    modelo = modelo.fit()
    ypred = modelo.get_forecast(len(Yprueba))
    y_pred_df = ypred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = modelo.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = Yprueba.index
    y_pred_out = y_pred_df["Predictions"]
    plt.plot(y_pred_out,color = 'purple', label = 'Predicciones con ARMA')



predRegresionLineal(Xprueba,Yprueba,Xentrenamiento,Yentrenamiento.values)
predARMA(Xprueba,Yprueba,Xentrenamiento,Yentrenamiento)
plt.legend()
plt.show()

