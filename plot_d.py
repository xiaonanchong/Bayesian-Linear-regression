import matplotlib 
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import answers as an

def Phi(N, M, X):
  phi = np.empty([N, M])
  for i in range(N):
    for j in range(M):
      if(j==0):
        phi[i][j]=1
      else:
        phi[i][j]=np.exp(-(X[i][0]-mu[j-1])**2 / (2*L**2) )
  return phi


def mean_sd(predict):
  l = predict.shape[0]
  s = 0
  for i in range(l):
    s = s+predict[i]
  mean = s/float(l)

  ss = 0
  for i in range(l):
    dif = (mean-predict[i])**2
    ss = ss + dif
  sd = (ss/float(l))**(0.5)
  return mean, sd


alpha = 1
beta = 0.1 
N = 25
M = 11

# TRAINING DATA
X = np.reshape(np.linspace(0,0.9,N),(N,1))
y = np.reshape(np.cos(10*X**2) + 0.1*np.sin(100*X),(N,1))

# MODEL order: M
mu = np.linspace(-0.5, 1, 10)#10 Gausian basis functions equally spaced between -0.5 and 1 
L = 0.1 #with scale 0.1
phi = Phi(N, M, X)
phi_t = np.transpose(phi)

# TESTING DATA
num = 300
t = np.linspace(-1, 1.5, num)#for drawing
Xt = np.reshape(np.linspace(-1, 1.5, num),(num, 1))
yt = np.reshape(np.cos(10*Xt**2) + 0.1*np.sin(100*Xt),(num,1))
F = Phi(num, M, Xt)#[300, 10]

nums = 45
t_sparse = np.linspace(-1, 1.5, nums)#for drawing
yt_sparse = np.reshape(np.cos(10*t_sparse**2) + 0.1*np.sin(100*t_sparse),(nums,1))



s0 = alpha * np.identity(M)
m0 = np.zeros((M, 1))

sn = np.linalg.inv((1/beta)*np.dot(phi_t, phi) + np.linalg.inv(s0))#[M,M]
mn = np.dot( sn, ((1/beta)*np.dot(phi_t, y) + np.dot(np.linalg.inv(s0), m0)))#[M,1]

## SAMPLE FROM POSTERIOR
mean = np.ndarray.flatten(mn)
cov = sn
size = 5
thetas = np.random.multivariate_normal(mean, cov, size)

## PREDICTION [num, 5]
pos_predict = np.zeros((num, 5))
for i in range(5):
  theta = thetas[i]
  pos_predict[:, i] = np.dot(F, theta)
#print(pos_predict[1].shape[0])#5

pos_pre_sta = np.zeros((num, 2))
for i in range(num):
  predict = pos_predict[i]
  mean, sd = mean_sd(predict)
  pos_pre_sta[i][0] = mean
  pos_pre_sta[i][1] = sd
#print(pos_pre_sta)

mean_curve = np.dot(F, mn)#pos_pre_sta[:,0] #[300,1]
#print(mean_curve.shape)#[300,1]
'''
u = mean_curve
b = mean_curve
V = np.dot(np.dot(F, sn), np.transpose(F))
for i in range(num):
  u[i][0] = u[i][0] + V[i][i]
  upper_curve = np.ndarray.flatten(u)
  b[i][0] = b[i][0] - V[i][i]
  bottom_curve = np.ndarray.flatten(b)
'''
V = np.zeros(num)
u = np.zeros((num,1))
b = np.zeros((num,1))
for i in range(num):
  phi = F[i]
  v = np.dot(np.dot(phi, sn), np.transpose(phi))
  print(v)
  u[i][0] = mean_curve[i][0] + v
  b[i][0] = mean_curve[i][0] - v
upper_curve = np.ndarray.flatten(u)
bottom_curve = np.ndarray.flatten(b)

#upper_curve = pos_pre_sta[:,0] + pos_pre_sta[:,1]
#bottom_curve = pos_pre_sta[:,0] - pos_pre_sta[:,1]

#print(upper_curve)
#print(bottom_curve)
#print(upper_curve == bottom_curve)


fig, ax = plt.subplots()
ax.set_ylim([-2.2,2.2])
plt.fill_between(t, upper_curve, bottom_curve, color = 'grey', alpha = 0.3)

plt.plot(t_sparse, yt_sparse, 'k+')
plt.plot(t, mean_curve, lw=2.5, color='k')
plt.plot(t, upper_curve+beta,'k--')
plt.plot(t, bottom_curve-beta,'k--')
for i in range(5):
  plt.plot(t, pos_predict[:, i], lw=1)


#'sample 1','sample 2','sample 3','sample 4','sample 5',
plt.legend(('true data', 'predictive mean', 'noise prediction'),
           loc='upper right', shadow=True)

ax.set_title('Bayesian Linear Regression')
ax.set_xlabel('input data x')
ax.set_ylabel('predicted value')
plt.show()

 
