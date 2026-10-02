# S7_Q1_BarChart.py

import matplotlib.pyplot as plt

courses = ["BCA", "BBA", "B.Com", "B.Sc", "BA"]
students = [60, 45, 75, 50, 35]

colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.bar(courses, students, color=colors)

plt.title("Students Enrolled in Different Courses")
plt.xlabel("Courses")
plt.ylabel("Number of Students")

plt.show()






# S7_Q2_KNN_Student.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [60, 65, 70, 75, 80, 82, 85, 90, 92, 95],
    'Performance': [
        'Low', 'Low', 'Low', 'Average', 'Average',
        'Average', 'Good', 'Good', 'Good', 'Good'
    ]
})

X = data[['Study_Hours', 'Attendance']]
y = data['Performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)





