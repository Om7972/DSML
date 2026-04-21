import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
from glob import glob
from sklearn.metrics import accuracy_score

dataset_frames = []
for csv_path in sorted(glob(os.path.join('datasets', '*.csv'))):
    frame = pd.read_csv(csv_path)
    if 'prognosis' in frame.columns:
        dataset_frames.append(frame)

if not dataset_frames:
    raise RuntimeError("No training datasets with 'prognosis' column were found in datasets/.")

dataset = pd.concat(dataset_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=20)

svc = SVC(kernel='linear', probability=True, random_state=20)
svc.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
with open('models/svc .pkl', 'wb') as f:
    pickle.dump(svc, f)
with open('models/svc.pkl', 'wb') as f:
    pickle.dump(svc, f)

y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model retrained successfully using {len(dataset_frames)} prognosis datasets.")
print(f"Total training rows: {len(dataset)}")
print(f"Holdout accuracy: {acc:.4f}")
