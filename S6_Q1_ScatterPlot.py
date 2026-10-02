# S6_Q1_ScatterPlot.py


import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Car_Model': ['Car A', 'Car B', 'Car C', 'Car D', 'Car E'],
    'Engine_Size': [1.2, 1.5, 2.0, 2.5, 3.0],
    'Fuel_Efficiency': [22, 20, 17, 14, 11]
})

print(data)

plt.scatter(data['Engine_Size'], data['Fuel_Efficiency'], c='blue')
plt.title("Engine Size vs Fuel Efficiency")
plt.xlabel("Engine Size (Litres)")
plt.ylabel("Fuel Efficiency (km/L)")
plt.show()




# S6_Q2_LinearRegression.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.DataFrame({
    'Training_Hours_Per_Week': [2, 4, 5, 7, 8, 10, 12, 14],
    'Performance_Score': [45, 50, 55, 62, 68, 75, 82, 90]
})

X = data[['Training_Hours_Per_Week']]
y = data['Performance_Score']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

print("R2 Score:", r2)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Training Hours Per Week")
plt.ylabel("Performance Score")
plt.title("Training Hours vs Performance Score")
plt.show()