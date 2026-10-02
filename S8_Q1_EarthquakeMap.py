# S8_Q1_EarthquakeMap.py


import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Latitude': [19.07, 28.61, 23.25, 34.05, 35.68, 13.08, 26.85, 31.63],
    'Longitude': [72.87, 77.20, 77.41, 118.25, 139.69, 80.27, 80.95, -8.00],
    'Depth': [10, 15, 20, 12, 25, 18, 30, 14],
    'Magnitude': [4.5, 5.2, 4.8, 6.1, 7.0, 4.2, 5.8, 6.5]
})

print("Earthquake Data:")
print(data)

plt.figure(figsize=(10, 6))

plt.scatter(
    data['Longitude'],
    data['Latitude'],
    c=data['Magnitude'],
    cmap='jet',
    s=data['Magnitude'] * 50
)

plt.colorbar(label='Magnitude')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geospatial Visualization of Earthquakes")

plt.show()






# S8_Q2_SalaryRegression.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.DataFrame({
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [30000, 35000, 40000, 45000, 52000, 58000, 65000, 72000, 80000, 90000]
})

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")
plt.show()