import numpy as np 
from scipy import random, linalg, dot, diag, all, allclose
import copy
import time
    

    

m=input("Enter number of rows: ")
n=input("Enter number of columns: ")


m=int(m)
n=int(n)

A = np.random.uniform(low=0.5, high=13.3, size=(m,n))
Original_A=copy.deepcopy(A)
print('\n')
print("Matrix generated:\n")

for i in range(m):
    for j in range(n):
        print '{:4}'.format(A[i][j]),
    print

print('\n')

def v_i_into_v_i_T(v_i,v_i_T,n):
	res=np.zeros([n,n],dtype=float)
	for i in range(n):
		for j in range(n):
			res[i][j]=(v_i[i]*v_i_T[j])
	return res






def getQ(A, betas):
	print(betas)
	
	m,n = A.shape
	Q = np.eye(m)
	lenbeta=n-len(betas)
	if len(betas)<n:
		for x in range(lenbeta):
			betas.append(0.0)
	
	for j in range(len(betas)-1,-1,-1):
		v=np.zeros(m)
		v[j]=1.0
		v[j+1:m]=copy.deepcopy(A[j+1:m,j])
		I=np.eye(m-j)
		x=copy.deepcopy(np.matmul((I-np.multiply(betas[j],v_i_into_v_i_T(v[j:m],v[j:m].T,len(v[j:m])))),Q[j:,j:]))		
		Q[j:,j:] =x
		
	return Q
   
def house(x):

	n = x.shape[0]
	norm1 = x[1:].dot(x[1:])
	v = np.copy(x)
	v[0] = 1.0
	if norm1 < np.finfo(float).eps:
		beta = 0.0
	else:
		norm_x= np.sqrt(x[0]**2 + norm1)
		if x[0] <= 0:
			v[0] = x[0] - norm_x
		else:
			v[0] = -norm1 / (x[0] + norm_x)
		beta = 2 * v[0]**2 / (norm1 + v[0]**2)
		v = v / v[0]
	return v, beta



betas=[]
def qr_column_pivoting(A,m,n):
	c=[]
	t=n-1
	betas=[]
	for j in range(n):
		c.append(np.dot(A[0:m,j].T,A[0:m,j]))
	r=0
	tou=max(c)
	
	k=0
	flag=0
	for k1 in range(n):
		if c[k1]==tou and flag==0:
			k=k1
			flag=1
	piv=[0]*n
	
	while tou>0:
		
		piv[r]=k
		temp=copy.deepcopy(A[0:m,r])
		A[0:m,r]=A[0:m,k]
		A[0:m,k]=temp
		temp1=copy.deepcopy(c[k])
		c[k]=c[r]
		c[r]=temp1
		v,beta=house(A[r:m,r])
		
		betas.append(beta)
		
		I=np.eye(m-r)
		A[r:m,r:n]=np.matmul((I-np.multiply(beta,v_i_into_v_i_T(v,v.T,len(v)))),A[r:m,r:n])
		if r<m:
			A[r+1:m,r]=v[1:m-r+1]

		for i in range(r+1,n):
			c[i]=c[i]-(A[r,i]*A[r,i])

		if r<t:
			tou=max(c[r+1:n])
			flag=0
			for k1 in range(r+1,n):
				if c[k1]==tou and flag==0:
					k=k1
					flag=1
			
		else:
			tou=0

		r+=1
	return betas

B=copy.deepcopy(A)
start = time.time()
betas=qr_column_pivoting(A,m,n)
print("\n")


print("final result: \n",A)
print("\n")

print("R \n",np.triu(A))
print("\n")

q5, r5, p5 = linalg.qr(B, overwrite_a=False,mode='economic', pivoting=True)
print("using scipy Q \n",q5)
print("\n")

print("using scipy R \n",r5)
print("\n")


print("Q my algo\n",getQ(A,betas))
end = time.time()
tim=(end-start)
print(tim," seconds")