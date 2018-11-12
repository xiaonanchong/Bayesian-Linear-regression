import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math 

import answers as an

Xd = np.linspace(0,0.9,25)
Yd = np.cos(10*Xd**2) + 0.1*np.sin(100*Xd)
phi = np.reshape(Xd, [25,1])
delta = 0.005
a = np.arange(0.1, 1.0, delta) #M
b = np.arange(0.1, 1.0, delta) #N
m = a.shape[0]
n = b.shape[0]
z = np.empty([n,m])
for a1 in range(m):
  for b1 in range(n):
    z[b1][a1] = an.lml(a[a1], b[b1], phi, Yd)
#print(z.shape)

fig, ax = plt.subplots()
CS = ax.contour(a,b,z,100)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Maximize Log marginal Likelihood - Linear Function Basis')
#(starting point-(0.9,0.9) step_size = 0.025)
ax.set_xlabel('alpha >= 0')
ax.set_ylabel('beta >= 0')


def plot2Dpoint(x,y):
  ax.hold(True)
  plt.scatter(x, y)

def grad(x,y):
  gama = 0.005
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
    
    if(mlml1-mlml == 0 or count > 1000): 
      print(mlml1, math.exp(mlml1), x, y)
      v = False

    mlml = mlml1
    count = count + 1

plot2Dpoint(0.9,0.9)
grad(0.9,0.9)
plt.show()
