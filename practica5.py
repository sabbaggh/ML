import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import copy

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
#Se hace una SVM con un kernel lineal y con uno contra todos
maquinaSoporte = svm.SVC(kernel='linear', decision_function_shape='ovr')

########################Definir funciones para el algoritmo genetico
#######################
######################
####################
####################
individuos = 10
caracteristicas = len(X[0])
#Se crea la poblacion inicial con numeros random 0 o 1
#Se le ponen cinco columnas ya que en la ultima se almacenara la precision y de esta manera sera mas facil manejar los datos
poblacionInicial = np.random.randint(2,size=(individuos,caracteristicas+1))
poblacionInicial = poblacionInicial.astype(float)
#print(poblacionInicial)


def evaluacion(poblacion):
    precisiones = np.zeros((individuos,k))
    for j in range(individuos):
        for i in range(k):
            Xprueba = X[i * tamFolds:(i + 1) * tamFolds, :]
            Xentrenamiento = np.delete(X, slice(i * tamFolds, (i + 1) * tamFolds), 0)
            Xentrenamiento = scaler.fit_transform(Xentrenamiento)
            Xprueba = scaler.transform(Xprueba)
            Xprueba2 = crearXprueba(Xprueba, poblacion, j)
            Yprueba = Y[i * tamFolds:(i + 1) * tamFolds]
            Yentrenamiento = np.delete(Y, slice(i * tamFolds, (i + 1) * tamFolds), 0)
            maquinaSoporte.fit(Xentrenamiento, Yentrenamiento)
            precisionSVM = maquinaSoporte.score(Xprueba2, Yprueba)
            precisiones[j,i] = precisionSVM
    for i in range(individuos):
        poblacion[i,4] = np.mean(precisiones[i,:])
    indices_ordenados = np.argsort(poblacion[:, 4])[::-1]
    poblacion = poblacion[indices_ordenados]
    print()
    print(poblacion)
    return poblacion

def crearXprueba(Xprueba,poblacion,individuo):
    for i in range(caracteristicas):
        if poblacion[individuo,i] == 0:
            Xprueba[:,i] = 0
    return Xprueba

def cruza(poblacion):
    longitud = len(poblacion)
    hijos = np.zeros((0, caracteristicas+1))
    for i in range(int(longitud/2)):
        padre1 = poblacion[i, :]
        padre2 = poblacion[(longitud-1)-i, :]
        puntoCruce = random.randint(0, caracteristicas-2)
        hijo1 = np.zeros((1, caracteristicas+1))
        hijo2 = np.zeros((1, caracteristicas+1))
        hijo1[0, :puntoCruce+1] = padre1[:puntoCruce+1]
        hijo1[0, puntoCruce+1:] = padre2[puntoCruce+1:]
        hijo2[0, :puntoCruce+1] = padre2[:puntoCruce+1]
        hijo2[0, puntoCruce+1:] = padre1[puntoCruce+1:]
        hijos = np.vstack([hijos, hijo1, hijo2])
    #Se resetean las precisiones a 0
    hijos[:,4] = 0
    return hijos

def mutacion(poblacion):
    genesMutacion = np.random.randint(4,size=individuos)
    print(genesMutacion)
    for i in range(individuos):
        if poblacion[i,genesMutacion[i]] == 1:
            poblacion[i, genesMutacion[i]] = 0
        else:
            poblacion[i, genesMutacion[i]] = 1
    return poblacion


poblacionEvaluada = evaluacion(poblacionInicial)
poblacionAMutar = copy.deepcopy(poblacionEvaluada)
poblacionMutada = mutacion(poblacionAMutar)
hijos = cruza(poblacionEvaluada)
print(poblacionEvaluada)
print(poblacionMutada)
print(hijos)

def algoritmoGenetico():
    diferencia = 100
    epsi = 0.00001
    while diferencia>epsi:

