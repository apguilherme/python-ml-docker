# run: python forest.py

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

# build model
clf = RandomForestClassifier(n_estimators=10)

# train classifier
clf.fit(X_train, y_train)

# predict
predicted = clf.predict(X_test)

# check accuracy
print("accuracy_score: "+ str(accuracy_score(predicted, y_test)))

# pickle to create a byte stream from object to be able to predict from it
import pickle
with open("./pickles/forest.pkl", "wb") as model_pkl:
    pickle.dump(clf, model_pkl)
