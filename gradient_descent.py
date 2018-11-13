import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math 

def lml(alpha, beta, Phi, Y):
  N = Phi.shape[0]
  M = Phi.shape[1]
  p1 = (-N/2.0)*np.log(2*np.pi) 
  tem = alpha*np.matmul(Phi, np.transpose(Phi)) 
  p2 = (-1/2.0)*np.log(np.linalg.det(tem + beta*np.identity(N)))
  p3 = (-1/2.0)*np.matmul(np.matmul(np.transpose(Y), np.linalg.inv(tem + beta*np.identity(N))), Y)
  return np.asscalar(p1+p2+p3)


def grad_lml(alpha, beta, Phi, Y):
  N = Phi.shape[0]
  M = Phi.shape[1]
  z = np.matmul(np.matmul(Phi, alpha*np.identity(M)), np.transpose(Phi)) + beta*np.identity(N)
  A = np.linalg.inv(z)
  p1 = np.dot(Phi, np.transpose(Phi))
  p2 = np.dot(np.dot(A, p1), A)
  tem = np.dot(np.dot(np.transpose(Y), p2), Y)
  alpha_grad = (-1/2.0)*(np.trace(np.dot(A, np.dot(Phi, np.transpose(Phi)))) - tem)
  beta_grad = (-1/2.0)*(np.trace(A) - np.dot(np.dot(np.transpose(Y), np.dot(A,A)), Y))
  return np.array([alpha_grad, beta_grad]).flatten()

def plot2Dpoint(x,y):
  ax.hold(True)
  plt.scatter(x, y)

##data
x=np.linspace(0, 0.9, 25)
y=np.cos(10*x**2) + 0.1*np.sin(100*x)
phi = np.reshape([[1]*25, x], [25,2])


##plot data
delta = 0.003
a = np.arange(0.1, 1.0, delta) #M
b = np.arange(0.1, 1.0, delta) #N
m = a.shape[0]
n = b.shape[0]
z = np.empty([n,m])
for a1 in range(m):
  for b1 in range(n):
    z[b1][a1] = lml(a[a1], b[b1], phi, y)

##plot configeration
fig, ax = plt.subplots()
CS = ax.contour(a,b,z,60)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Maximize Log marginal Likelihood - Linear Function Basis')
ax.set_xlabel('alpha >= 0')
ax.set_ylabel('beta >= 0')

plt.show()
