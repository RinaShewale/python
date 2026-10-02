# S3_Q1_BoxPlot.py

import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Fuel_Efficiency': [18, 20, 22, 19, 60],
    'Engine_Power': [100, 110, 120, 115, 300],
    'Vehicle_Weight': [1200, 1250, 1300, 1280, 1400],
    'Engine_Size': [1200, 1400, 1600, 1500, 2000],
    'Acceleration': [12, 11, 10, 11.5, 8]
})

print(data)

plt.boxplot(data['Fuel_Efficiency'])
plt.title("Fuel Efficiency")
plt.ylabel("Fuel Efficiency")
plt.show()

plt.boxplot(data['Engine_Power'])
plt.title("Engine Power")
plt.ylabel("Engine Power")
plt.show()






# S3_Q2_RandomForest.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    'Age': [45, 50, 60, 35, 40, 65, 55, 30, 70, 48],
    'Cholesterol': [220, 240, 280, 180, 190, 300, 250, 170, 310, 230],
    'BloodPressure': [140, 150, 160, 120, 125, 170, 155, 110, 175, 145],
    'HeartDisease': [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
})

X = data[['Age', 'Cholesterol', 'BloodPressure']]
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)