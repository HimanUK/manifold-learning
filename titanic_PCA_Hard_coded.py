#%% Data Manipulation

import pandas as pd
import numpy as np

df = pd.read_csv('D:\Study\ML and ANN\Manifold Learning\Train.csv')
#test = pd.read_csv('D:\Study\ML and ANN\Manifold Learning\Test.csv')

df.drop(['PassengerId','Ticket','Cabin','Name'],axis =1, inplace = True)
df['Age'].fillna(df['Age'].mean(),inplace=True)
for a in df['Age']:
    if a<=16:
        df['Sex'] ='child'

df['Sex']=df['Sex'].map(dict(zip(('male','female','child'),(0,1,2))))
df['Embarked'].fillna('S', inplace=True)
df['Embarked']=df['Embarked'].map(dict(zip(('S','Q','C'),(0,1,2))))
    
#%% PCA starts
mean_vec = []
for i in df.columns:
    mean_vec.append(df[i].mean())

nd_train = df.values

for i in range(len(df.columns)):
    nd_train[:,i] = nd_train[:,i] - mean_vec[i]

cov_mat = np.cov(nd_train.T)

eig_val, eig_vec = np.linalg.eig(cov_mat)

#converting the data into new axes without manipulation
print(eig_val)

    

