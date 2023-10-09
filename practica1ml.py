import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

inicial = np.array([[1], [2], [3], [4], [5]])
beta = 0.5

x = np.empty([5,100])
'''for i in range(10):
    beta = beta+2*rng.normal()
    x[:,10*i:10*(i+1)] = (i+1)*inicial+rng.normal(size=(5,10))
    #np.insert(x,[i:10*i],xd,axis=1)
#print (x)'''
for i in range(10):
    beta = beta+2*rng.normal()
    x[:,10*i:10*(i+1)] = (i+1)*inicial+rng.normal(size=(5,10))
    #np.insert(x,i,xd,axis=1)

y = x[4,:]
y = y.T
x = x[0:4,:]
xT = x.T
Q = np.dot(x,xT)
b = np.dot(-2*x,y)
w0 = rng.normal(size=(4,1))
alfa = 0.00001
ep = 1
plt.scatter(x[0,:],x[1,:])
plt.show()
'''while ep > alfa:
    G = np.dot(2*Q,w0)+b
    wn = w0-alfa*G
    xddd = wn-w0
    ep = np.linalg.norm(xddd)
    w0 = wn
    print(wn)'''




