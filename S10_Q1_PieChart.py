# S10_Q1_PieChart.py


import matplotlib.pyplot as plt

transport = ["Train", "Bicycle", "Bike", "Car", "Bus"]
employees = [220, 60, 70, 80, 40]

colors = ["red", "blue", "green", "orange", "purple"]

plt.pie(
    employees,
    labels=transport,
    colors=colors,
    autopct="%1.1f%%"
)

plt.title("Preferred Mode of Transportation")
plt.show()






# S10_Q2_RandomForestRegression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a dummy housing.csv file since it is missing in the workspace
dummy_data = pd.DataFrame({
    'Area': [1500, 2000, 1800, 2400, 3000, 1200, 1700, 2200, 2800, 1600],
    'Bedrooms': [3, 4, 3, 4, 5, 2, 3, 3, 4, 3],
    'Bathrooms': [2, 2.5, 2, 3, 3.5, 1, 2, 2.5, 3, 2],
    'Age': [10, 5, 15, 8, 2, 20, 12, 6, 4, 14],
    'Price': [250000, 350000, 280000, 410000, 520000, 180000, 260000, 340000, 480000, 240000]
})
dummy_data.to_csv("housing.csv", index=False)

# Load the dataset
data = pd.read_csv("housing.csv")

print("Original Data:")
print(data.head())

data = data.dropna()

X = data[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)