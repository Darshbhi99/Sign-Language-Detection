import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data_dict = pkl.load(open("data.pkl", "rb"))

data = np.asarray(data_dict['data'])
label = np.asarray(data_dict['label'])

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=label)

model = RandomForestClassifier()

model.fit(X_train, y_train)

X_pred = model.predict(X_train)

y_pred = model.predict(X_test)

Train_score = accuracy_score(y_train, X_pred)

Test_score = accuracy_score(y_test, y_pred)

print(f"Accuracy of the Model for Training Data : {Train_score*100} !!!!")
print(f"Accuracy of the Model for Test Data : {Test_score*100} !!!!")

with open("model.pkl", 'wb') as f:
    pkl.dump(model, f)
f.close()