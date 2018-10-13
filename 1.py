import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
M= pd.read_csv("train.csv")

df = pd.DataFrame(M)

df = df[np.isfinite(df['Age'])]
df = df[np.isfinite(df['Fare'])]
columns_for_differencing = ['PassengerId','Sex','Name','Ticket','Cabin','Embarked']
X_train= df.copy()[df.columns.difference(columns_for_differencing)]
X_traind=X_train[X_train.Survived == 0]
X_trains=X_train[X_train.Survived == 1]
X_traind=X_traind.values
X_trains=X_trains.values
X_traind1=X_traind[:,0:4]
X_trains1=X_trains[:,0:4]


# print(len(X_traind1))
c = np.array([[0.00] * 4] *714)
v = np.array([[0.0000]*4]*3)



for i in range(0,290):
    for j in range(4):
        c[i][j]=X_trains1[i][j]

for i in range(290,714):
    for j in range(4):
        c[i][j]=X_traind1[i-290][j]

#print("reduced data set with sepal length/width, petal length/width")
print(c)   
c=c.transpose()
Mcov=np.cov(c)
#print("\ncovariance of given matrix: \n",Mcov,"\n")
E,V=np.linalg.eig(Mcov)#E=eigen value,V=eigen vector
#print("eigen values are: \n",E,"\n")
#print("eigen vectors are: \n",V,"\n")
V=V.transpose()
for i in range(3):
    for j in range(4):
        v[i][j]=V[i][j]
#print("one vector chosen for PCA:\n",v,"\n")
M2=np.matmul(v,c)
M2=M2.transpose()
print(M2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(M2[0:290,0],M2[0:290,1],M2[0:290,2])
ax.scatter(M2[290:714,0],M2[290:714,1],M2[290:714,2])
plt.show()

