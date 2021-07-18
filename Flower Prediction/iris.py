import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('iris.data')

x=df[['A','B','C','D']]
y=df['E']

print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
le.transform(y)

from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(x_train, y_train)

pickle.dump(sv, open('iris.pkl', 'wb'))