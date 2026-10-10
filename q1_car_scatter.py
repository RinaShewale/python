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

plt.scatter(df["Engine Size"], df["Fuel Efficiency"],
            color=["red", "blue", "green", "orange", "purple"])

plt.title("Engine Size vs Fuel Efficiency")
plt.xlabel("Engine Size (Litres)")
plt.ylabel("Fuel Efficiency (km/L)")

plt.show()





# S1_Q2_Training.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Sports_Performance.csv")

X = df[["Training Hours Per Week"]]
y = df["Performance Score"]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

print("R2 Score:", r2_score(y, y_pred))

plt.scatter(X, y)
plt.plot(X, y_pred)

plt.title("Training Hours vs Performance Score")
plt.xlabel("Training Hours Per Week")
plt.ylabel("Performance Score")

plt.show()