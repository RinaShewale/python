import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Fuel Efficiency": [18, 20, 22, 19, 45],
    "Engine_Power": [100, 120, 110, 130, 300],
    "Vehicle_Weight": [1200, 1300, 1250, 1400, 1500],
    "Engine_Size": [1.2, 1.5, 1.4, 1.8, 2.0],
    "Acceleration": [10, 9, 11, 8, 7]
}

df = pd.DataFrame(data)

print(df)

plt.boxplot(df["Fuel Efficiency"])
plt.title("Box Plot of Fuel Efficiency")
plt.ylabel("Fuel Efficiency")
plt.show()

plt.boxplot(df["Engine_Power"])
plt.title("Box Plot of Engine Power")
plt.ylabel("Engine Power")
plt.show()







# S8_Q2_Data_Transformation.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = {
    "Temperature": [20, 25, 30, 35, 40],
    "Humidity": [40, 50, 60, 70, 80],
    "Wind Speed": [5, 10, 15, 20, 25],
    "Pressure": [1000, 1010, 1020, 1030, 1040],
    "Rainfall": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

minmax = MinMaxScaler()
standard = StandardScaler()

minmax_data = minmax.fit_transform(df)
standard_data = standard.fit_transform(df)

minmax_df = pd.DataFrame(minmax_data, columns=df.columns)
standard_df = pd.DataFrame(standard_data, columns=df.columns)

print("Original Data:")
print(df)

print("\nMin-Max Scaled Data:")
print(minmax_df)

print("\nStandardized Data:")
print(standard_df)