import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Cargar dataset
df = pd.read_csv('6 class csv2.csv')
#Revolver las instancias del dataset
df = df.sample(frac=1)
#Numero de folds
k = 4
#se pasa a una matriz el dataset
datos = df.to_numpy()
tam = len(datos)
#Se calcula el tamanio de cada fold
tamFolds = int(tam/k)
#Se define X
X = datos[:,0:4]
#Se define Y
Y = datos[:,4]
Y = Y.astype(int)
scaler = StandardScaler()
#Variables para calcular el error total
errorSVM = 0
errorregLog = 0
errorBayes = 0
errorRed = 0
#Se hace una SVM con un kernel lineal y con uno contra todos
maquinaSoporte = svm.SVC(kernel='linear', decision_function_shape='ovr')
#Se hace una regresion logistica, igual multiclase con uno contra todos y se define un maximo de iteraciones ya que a veces
#no llega a converger o se requieren muchas iteraciones
regLog = LogisticRegression(multi_class='ovr', max_iter=1000)
#Se define un clasificador bayesiano, a este no se le tienen que ajustar los parametros
bayes = GaussianNB()
#Se define la red neuronal, definimos el numero de capas ocultas
redNeuronal = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 5), random_state=1, max_iter=1000)
#se hace un ciclo for para el numero de folds
for i in range(k):
    #Se definen los valores de prueba y de entrenamiento y se normalizan
    Xprueba = X[i*tamFolds:(i+1)*tamFolds,:]
    Xentrenamiento = np.delete(X,slice(i*tamFolds,(i+1)*tamFolds),0)
    Xentrenamiento = scaler.fit_transform(Xentrenamiento)
    Xprueba = scaler.transform(Xprueba)
    Yprueba = Y[i*tamFolds:(i+1)*tamFolds]
    Yentrenamiento = np.delete(Y, slice(i*tamFolds,(i+1)*tamFolds),0)
    #se ajusta el modelo de SVM a nuestros valores de entrenamiento
    maquinaSoporte.fit(Xentrenamiento,Yentrenamiento)
    #Se ajusta el modelo de regresion logistica a nuestros valores de entrenamiento
    regLog.fit(Xentrenamiento,Yentrenamiento)
    #se ajusta el modelo bayesiano a nuestros valores de entrenamiento
    bayes.fit(Xentrenamiento,Yentrenamiento)
    #Se ajusta la red neuronal a nuestros valores de entrenamiento
    redNeuronal.fit(Xentrenamiento,Yentrenamiento)
    #Se calcula la precision de cada modelo
    precisionSVM = maquinaSoporte.score(Xprueba,Yprueba)
    precisionRegLog = regLog.score(Xprueba, Yprueba)
    precisionBayes = bayes.score(Xprueba, Yprueba)
    precisionRed = redNeuronal.score(Xprueba,Yprueba)
    #Se actualiza el error total
    errorSVM = errorSVM + 1 - precisionSVM
    errorregLog = errorregLog + 1 - precisionRegLog
    errorBayes = errorBayes + 1 - precisionBayes
    errorRed = errorRed + 1 - precisionRed
    #Se imprime la precision de cada modelo en cada fold
    print(f'Precision de SVM en el fold {i+1} {precisionSVM*100}%')
    print(f'Precision de Regresion Logistica en el fold {i+1} {precisionRegLog*100}%')
    print(f'Precision de Bayes en el fold {i+1} {precisionBayes*100}%')
    print(f'Precision de la Red Neuronal en el fold {i + 1} {precisionRed * 100}%')
    print()

errorPromSVM = (errorSVM/k)*100
errorPromRegLog = (errorregLog/k)*100
errorPromBayes = (errorBayes/k)*100
errorPromRed = (errorRed/k)*100
print(f'Error promedio en SVM {errorPromSVM}%')
print(f'Error promedio en Regresion Logistica {errorPromRegLog}%')
print(f'Error promedio en clasificador Bayesiano {errorPromBayes}%')
print(f'Error promedio en Red Neuronal {errorPromRed}%')
