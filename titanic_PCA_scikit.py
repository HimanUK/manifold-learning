import pandas as pd
import numpy as np

train = pd.read_csv('D:\Study\ML and ANN\Manifold Learning\Train.csv')
test = pd.read_csv('D:\Study\ML and ANN\Manifold Learning\Test.csv')

for df in train,test:
    df.drop('Name',axis =1, inplace = True)
    df.drop('Cabin',axis =1, inplace = True)
    df.drop('Ticket',axis =1, inplace = True)

    df['Age'].fillna(df['Age'].mean(),inplace=True)
    for a in df['Age']:
        if a<=16:
            df['Sex'] ='child'
    
    df['Sex']=df['Sex'].map(dict(zip(('male','female','child'),(0,1,2))))
    df['Embarked'].fillna('S', inplace=True)
    df['Embarked']=df['Embarked'].map(dict(zip(('S','Q','C'),(0,1,2))))
    
'''
colmean = []

for i in list(train):
    colmean.append(train[i].mean())
    for p in train[i]:
        p = p - colmean[-1]
'''

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

T = train.values
T = scale(T)

pca = PCA(n_components = 2)

pca.fit(T)

var = pca.explained_variance_ratio_
print(var)
var1 = np.cumsum(np.round(pca.explained_variance_ratio_,decimals = 4)*100)

print(var1)

plt.plot(var1)