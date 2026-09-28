import matplotlib.pyplot as plt
from numpy import array, zeros
def F(U):
    return(array([U[1], -U[0]]))
N=1000000
At=0.1
Nv=2
U=zeros((N+1,Nv))
U[0,:]=array([1,0])
for n in range(0,N):
    k1=F(U[n,:])
    k2=F(U[n,:]+At*k1/2)
    k3=F(U[n,:]+At*k2/2)
    k4=F(U[n,:]+At*k3)
    U[n+1,:]=U[n,:]+At*(k1+2*k2+2*k3+k4)/6
plt.plot(U[:,0],U[:,1])
plt.axis('equal')
plt.show()