import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import answers as an

Xd = np.linspace(0,0.9,25)
Yd = np.cos(10*Xd**2) + 0.1*np.sin(100*Xd)

def phi(N, M):
  X = np.linspace(0,0.9,N)
  phi = np.empty([N, M])
  for i in range(N):
    for j in range(M):
      if(j==0):
        phi[i][j]=1
      elif(j%2 ==1):
        J = (j+1)/2
        phi[i][j]=np.sin(2*np.pi*J*X[i])
      else:
        J = j/2
        phi[i][j]=np.cos(2*np.pi*J*X[i])
  return phi

def phi2(N, order):
  X = np.linspace(0,0.9,N)
  phi = np.empty([N, 2*order+1])
  for i in range(N):
    for j in range(2*order+1):
      if(j==0):
        phi[i][j]=1
      elif(j%2 ==1):
        J = (j+1)/2
        phi[i][j]=np.sin(2*np.pi*J*X[i])
      else:
        J = j/2
        phi[i][j]=np.cos(2*np.pi*J*X[i])
  return phi

phi = phi2(25,9)# order!

delta = 0.005#0.005[1-4]
a = np.arange(0.01, 0.5, delta) #M
b = np.arange(0.01, 0.5, delta) #N
m = a.shape[0]
n = b.shape[0]
z = np.empty([n,m])
for a1 in range(m):
  for b1 in range(n):
    z[b1][a1] = an.lml(a[a1], b[b1], phi, Yd)
#print(z.shape)

fig, ax = plt.subplots()
CS = ax.contour(a,b,z,50)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Maximize Log marginal Likelihood - Linear Function Basis')
#(starting point-(0.9,0.9) step_size = 0.025)
ax.set_xlabel('alpha >= 0')
ax.set_ylabel('beta >= 0')


def plot2Dpoint(x,y):
  ax.hold(True)
  plt.scatter(x, y)

def grad(x,y):
  gama = 0.00005
  v = True
  count = 0
  mlml = 0
  while(v==True):
  #for i in range(70):
    s1=x
    s2=y
    #print('s1 :',s1)
    #print('s2 :',s2)
    g1=an.grad_lml(x, y, phi, Yd)[0]
    g2=an.grad_lml(x, y, phi, Yd)[1]
    x=x+(gama)*g1
    y=y+(gama)*g2 
    t1=(gama)*g1
    t2=(gama)*g2 
    #print('t1 :',t1)
    #print('t2 :',t2)
    ax.arrow(s1,s2,t1,t2, head_width=0.009, head_length=0.01, fc='k', ec='k')
    #if(i==49):
      #plot2Dpoint(x,y)

    mlml1 = an.lml(x, y, phi, Yd)
    
    if(mlml1-mlml == 0 or count > 10000):
      if(mlml1-mlml < 0.00001 ):
        print(mlml1, x, y)
      v = False

    mlml = mlml1
    count = count + 1

  #return mlml


#plot2Dpoint(0.2,0.2)
grad(0.25,0.2)
plt.show()

# order = [0,4]
# [-28.4944341114782, -27.8019017751, -27.5613740577, -18.2780086987, -19.435735408, 
# order = [6, ]
# -9.075980021929912, 

