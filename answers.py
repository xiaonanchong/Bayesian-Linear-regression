# -*- coding: utf-8 -*-
"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""
import numpy as np

def lml(alpha, beta, Phi, Y):
  """
  4 marks

  :param alpha: float
  :param beta: float
  :param Phi: array of shape (N, M)
  :param Y: array of shape (N, 1)
  :return: the log marginal likelihood, a scalar
  """
  N = Phi.shape[0]
  M = Phi.shape[1]

  p1 = (-N/2.0)*np.log(2*np.pi)
  
  tem = alpha*np.matmul(Phi, np.transpose(Phi)) 
  p2 = (-1/2.0)*np.log(np.linalg.det(tem + beta*np.identity(N)))
  p3 = (-1/2.0)*np.matmul(np.matmul(np.transpose(Y), np.linalg.inv(tem + beta*np.identity(N))), Y)

  return np.asscalar(p1+p2+p3)


def grad_lml(alpha, beta, Phi, Y):
  """
  8 marks (4 for each component)

  :param alpha: float
  :param beta: float
  :param Phi: array of shape (N, M)
  :param Y: array of shape (N, 1)
  :return: array of shape (2,). The components of this array are the gradients
  (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.
  """

  N = Phi.shape[0]
  M = Phi.shape[1]

  z = np.matmul(np.matmul(Phi, alpha*np.identity(M)), np.transpose(Phi)) + beta*np.identity(N)

  A = np.linalg.inv(z)
  
  p1 = np.dot(Phi, np.transpose(Phi))
  p2 = np.dot(np.dot(A, p1), A)
  tem = np.dot(np.dot(np.transpose(Y), p2), Y)
  #print(tem)

  alpha_grad = (-1/2.0)*(np.trace(np.dot(A, np.dot(Phi, np.transpose(Phi)))) 
- tem)

  beta_grad = (-1/2.0)*(np.trace(A) - np.dot(np.dot(np.transpose(Y), np.dot(A,A)), Y))
  

  return np.array([alpha_grad, beta_grad]).flatten()

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

Xd = np.linspace(0,0.9,25)
Yd = np.cos(10*Xd**2) + 0.1*np.sin(100*Xd)
#print(lml(0.27, 0.17, phi(25,3), Yd))
#print(grad_lml(1.0, 0.3, np.reshape(Xd, (25,1)), Yd)[0]) #phi(25, 0)


