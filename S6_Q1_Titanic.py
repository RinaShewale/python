import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

plt.hist(df["Pclass"], bins=3, edgecolor="black")

plt.title("Number of Passengers in Each Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.show()

survival = df["Survived"].value_counts()

plt.pie(
    survival,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%"
)

plt.title("Survival Distribution")

plt.show()




# S6_Q2_Spam_KNN.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv("spambase.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:")
print(cm)