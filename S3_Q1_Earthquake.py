# S3_Q1_Earthquake.py

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Latitude": [19.07, 28.61, 23.02, 34.05, 13.08],
    "Longitude": [72.87, 77.20, 72.57, -118.24, 80.27],
    "Depth": [10, 15, 20, 12, 18],
    "Magnitude": [4.5, 5.2, 6.1, 7.0, 4.8]
}

df = pd.DataFrame(data)

plt.scatter(
    df["Longitude"],
    df["Latitude"],
    c=df["Magnitude"],
    cmap="Reds",
    s=100
)

plt.colorbar(label="Magnitude")
plt.title("Earthquake Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()




# S3_Q2_Salary_Regression.py


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Salary.csv")

X = df[["YearsExperience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red")

plt.title("Salary Prediction using Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.show()