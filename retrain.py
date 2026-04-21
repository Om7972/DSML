import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

dataset = pd.read_csv('datasets/Training.csv')
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=20)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

with open('models/svc .pkl', 'wb') as f:
    pickle.dump(svc, f)
print("Model retrained successfully with LabelEncoded target values.")
