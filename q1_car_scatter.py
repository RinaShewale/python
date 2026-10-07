# S1_Q1_Car.py

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Car Model": ["Car A", "Car B", "Car C", "Car D", "Car E"],
    "Engine Size": [1.2, 1.5, 1.8, 2.0, 2.5],
    "Fuel Efficiency": [20, 18, 16, 14, 11]
}

df = pd.DataFrame(data)

print(df)

plt.scatter(df["Engine Size"], df["Fuel Efficiency"], color="blue")

plt.title("Engine Size vs Fuel Efficiency")
plt.xlabel("Engine Size (Litres)")
plt.ylabel("Fuel Efficiency (km/L)")

plt.show()







# S1_Q2_Customer_KNN.py


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Customer.csv")

X = df[["Age", "Annual Income"]]
y = df["Segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))