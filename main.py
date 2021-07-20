# import statements
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

import my_methods as mm

# reading csv file for dataset
file_name = "iris.csv"
df = pd.read_csv(file_name)

# setting up data into x and y
cols = df.columns
x_cols = cols[:4]
y_cols = cols[4]

X = df[x_cols]
y = df[y_cols]

train = 0.8
test = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train, test_size=test)

# random forest
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
score = model.score(X_test, y_test)
print(mm.to_percent(score))

# displaying confusion matrix as a heatmap
cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
