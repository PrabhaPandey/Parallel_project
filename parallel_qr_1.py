import numpy as np 
from scipy import random, linalg, dot, diag, all, allclose
import copy
import time
from mpi4py import MPI
import math
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
starttime = MPI.Wtime()

def v_i_into_v_i_T(v_i,v_i_T,n):
	res=np.zeros([n,n],dtype=float)
	for i in range(n):
		for j in range(n):
			res[i][j]=(v_i[i]*v_i_T[j])
	return res

def getQ(A1, betas):
	print("beta values: ",betas)
	print('\n')
	A=[]
	for i in A1:
		A.append(i.tolist())
	A=np.array(A)
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
	if n==0:
		return x,0.0
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

def qr_column_pivoting(A1,m,n):
	c=[]
	t=n-1
	betas=[]
	A=[]
	for i in A1:
		A.append(i.tolist())
	
	A=np.array(A)
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
	return betas,A


def split_A(A,m,n):
	x=math.ceil(m/(size-1))
	x=int(x)
	if m%(size-1)!=0:
		x+=1
	L=[]
	i=0
	while(i<m-1):
		temp=[]
		for j in range(x):
			temp.append(A[i])
			if i==m-1:
				break
			i+=1
		L.append(temp)
		if i==m-1:
			break
	return L


if rank==0:
	m=input()
	n=input()


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
	submatrices=split_A(A,m,n)
	R_matrices=[]
	Q_gathered=[]
	for i in range(size):
		R_matrices.append([])
		Q_gathered.append([])

	for i in range(1,size):
		comm.send(submatrices[i-1],dest=i,tag=i)
		comm.send(R_matrices[i-1],dest=i,tag=i)


comm.Barrier()

if rank!=0:
	matrix=comm.recv(source=0,tag=rank)
	B=copy.deepcopy(matrix)
	r=comm.recv(source=0,tag=rank)
	print "From processor ", rank," received matrix ",matrix," received r ",r 
	print '\n'
	m1,n1=np.array(matrix).shape
	betas,matrix=qr_column_pivoting(matrix,m1,n1)
	print "final result for processor ", rank, " is ",matrix
	print "\n"
	R=np.triu(matrix)
	print "R for processor ", rank, " is ",R
	print "\n"

	Q_got=getQ(matrix,betas)
	print "Q for processor ", rank, " is ",Q_got
	print "\n" 

	q5, r5, p5 = linalg.qr(B, overwrite_a=False,mode='economic', pivoting=True)
	print "using scipy Q for processor ", rank, " is : ",q5,'\n'
	print "using scipy R for processor ", rank, " is : ",r5,'\n'
	
	comm.send(R,dest=0,tag=rank)
	comm.send(Q_got,dest=0,tag=rank)

comm.Barrier()

if rank==0:
	for i in range(1,size):
		x=comm.recv(source=i,tag=i)
		R_matrices[i-1]=x.tolist()
	
	
	for i in range(1,size):
		x=comm.recv(source=i,tag=i)
		Q_gathered[i-1]=x.tolist()

	R=[]
	for i in R_matrices:
		for j in i:
			R.append(j)
	
	R=np.array(R)
	B=copy.deepcopy(R)
	m2,n2=np.array(R).shape
	betas,R1=qr_column_pivoting(R,m2,n2)
	print "final result for gathered R is ",R1
	print "\n"
	R_=np.triu(R1)
	print "R_ for R is "
	print R_
	print "\n"

	Q_got=getQ(R1,betas)

	mq,nq=Q_got.shape
	Q_for_R=split_A(Q_got,mq,nq)

	
	Q_gathered=Q_gathered[:-1]
	
	final_Q=[]
	for i in range(len(Q_gathered)):
		final_Q.append(np.dot(Q_gathered[i],Q_for_R[i]).tolist())

	Q_final=[]
	for i in final_Q:
		for j in i:
			Q_final.append(j)
	print "final Q"
	for i in Q_final:
		for j in i:
			print j,
		print 
	print


	endtime   = MPI.Wtime()
	print "That took ",endtime-starttime," seconds for ", rank ," processor\n "