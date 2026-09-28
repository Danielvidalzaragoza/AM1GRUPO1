import matplotlib.pyplot as plt
from numpy import array, zeros 
from numpy.linalg import norm
def F(U):
    return(array([U[1], -U[0]]))
N=100
At=0.1
Nv=2
U=zeros((N+1,Nv))
U[0,:]=array([1,0])
for n in range(0,N):
    Y=U[n,:]
    while norm(Y-U[n,:]-At/2*(F(U[n,:])+F(Y)))>1e-6:
        R=Y-U[n,:]-At/2*(F(U[n,:])+F(Y))
        Y=Y+R
        print(norm(R))
        input("Press Enter to continue...")
    U[n+1,:]=Y
plt.plot(U[:,0],U[:,1])
plt.axis('equal')
plt.show()
