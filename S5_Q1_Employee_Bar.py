import matplotlib.pyplot as plt

departments = ["HR", "Finance", "IT", "Sales", "Marketing"]
employees = [20, 25, 40, 35, 30]

plt.bar(departments, employees, color=["red", "blue", "green", "orange", "purple"])

plt.title("Employees in Different Departments")
plt.xlabel("Departments")
plt.ylabel("Number of Employees")

plt.show()





# S5_Q2_Smartphone_Logistic.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

data = {
    "RAM": [4, 6, 8, 4, 8, 6, 12, 4, 8, 12],
    "Storage": [64, 128, 256, 64, 256, 128, 512, 64, 256, 512],
    "Battery": [4000, 4500, 5000, 4000, 5000, 4500, 5500, 4000, 5000, 5500],
    "Price": ["Low", "Medium", "High", "Low", "High",
              "Medium", "High", "Low", "High", "High"]
}

df = pd.DataFrame(data)

X = df[["RAM", "Storage", "Battery"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))