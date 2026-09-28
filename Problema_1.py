import matplotlib.pyplot as plt
from numpy import array, zeros
def F(U):
    return(array([U[1], -U[0]]))
N=10000
At=0.001
Nv=2
U=zeros((N+1,Nv))
U[0,:]=array([1,0])
for n in range(0,N):
    U[n+1,:]= U[n,:]+At*F(U[n,:])
plt.plot(U[:,0],U[:,1])
plt.axis('equal')
plt.show()