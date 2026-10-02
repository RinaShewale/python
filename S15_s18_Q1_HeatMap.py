# S15_Q1_HeatMap.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'Attendance': [60, 65, 70, 75, 80, 82, 88, 92],
    'Assignment_Score': [50, 55, 60, 65, 70, 75, 82, 90],
    'Exam_Score': [45, 50, 58, 62, 68, 74, 82, 88]
})

print("Student Performance Data:")
print(data)

correlation = data.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heat Map of Student Performance")
plt.show()








# S15_Q2_DataTransformation.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

data = pd.DataFrame({
    'Temperature': [20, 25, 30, 35, 40],
    'Humidity': [40, 50, 60, 70, 80],
    'WindSpeed': [5, 10, 15, 20, 25],
    'Pressure': [1000, 1005, 1010, 1015, 1020],
    'Rainfall': [10, 20, 30, 40, 50]
})

print("Original Data:")
print(data)

minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(data)

print("\nMin-Max Scaling:")
print(pd.DataFrame(minmax_data, columns=data.columns))

standard = StandardScaler()
standard_data = standard.fit_transform(data)

print("\nStandardization:")
print(pd.DataFrame(standard_data, columns=data.columns))

normalizer = Normalizer()
normal_data = normalizer.fit_transform(data)

print("\nNormalization:")
print(pd.DataFrame(normal_data, columns=data.columns))







# S18_Q1_BoxPlot.py
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Fuel_Efficiency': [18, 20, 22, 19, 60],
    'Engine_Power': [100, 110, 120, 115, 300],
    'Vehicle_Weight': [1200, 1250, 1300, 1280, 1400],
    'Engine_Size': [1200, 1400, 1600, 1500, 2000],
    'Acceleration': [12, 11, 10, 11.5, 8]
})

print("Vehicle Data:")
print(data)

# Fuel Efficiency Box Plot
plt.boxplot(data['Fuel_Efficiency'])
plt.title("Fuel Efficiency")
plt.ylabel("Fuel Efficiency")
plt.show()

# Engine Power Box Plot
plt.boxplot(data['Engine_Power'])
plt.title("Engine Power")
plt.ylabel("Engine Power")
plt.show()