# S17_Q1_DataCleaning.py


import pandas as pd

employee_data = pd.DataFrame({
    'Emp_ID': [101, 102, 103, 104, 105, 105, 107, 108],
    'Age': [25, 30, None, 45, 28, 28, 150, 35],
    'Dept': ['IT', 'HR', 'Sales', 'IT', 'Finance', 'Finance', 'HR', None],
    'Years_Experience': [2, 5, 3, 10, 4, 4, 20, None],
    'Performance_Rating': [4, 3, 5, 4, None, None, 2, 4]
})

print("Original Data:")
print(employee_data)

# Handling missing values
employee_data['Age'] = employee_data['Age'].fillna(
    employee_data['Age'].median()
)

employee_data['Dept'] = employee_data['Dept'].fillna(
    employee_data['Dept'].mode()[0]
)

employee_data['Years_Experience'] = employee_data['Years_Experience'].fillna(
    employee_data['Years_Experience'].median()
)

employee_data['Performance_Rating'] = employee_data['Performance_Rating'].fillna(
    employee_data['Performance_Rating'].median()
)

# Remove duplicate records
employee_data = employee_data.drop_duplicates()

# Detect outliers using IQR
Q1 = employee_data['Age'].quantile(0.25)
Q3 = employee_data['Age'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = employee_data[
    (employee_data['Age'] < lower) |
    (employee_data['Age'] > upper)
]

print("\nOutliers:")
print(outliers)

print("\nCleaned Data:")
print(employee_data)










# S17_Q2_WorldCities.py


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

data = pd.DataFrame({
    'City': ['Mumbai', 'London', 'New York', 'Tokyo', 'Delhi', 'Paris'],
    'Country': ['India', 'UK', 'USA', 'Japan', 'India', 'France'],
    'Latitude': [19.07, 51.50, 40.71, 35.68, 28.61, 48.85],
    'Longitude': [72.87, -0.12, -74.00, 139.69, 77.20, 2.35],
    'Population': [20000000, 9000000, 8400000, 14000000, 19000000, 2100000]
})

print("World Cities Data:")
print(data)

# Geospatial Visualization

plt.figure(figsize=(10, 6))

plt.scatter(
    data['Longitude'],
    data['Latitude'],
    s=data['Population'] / 50000
)

for i in range(len(data)):
    plt.text(
        data['Longitude'][i],
        data['Latitude'][i],
        data['City'][i]
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("World Cities Geospatial Visualization")
plt.grid()

plt.show()


# Treemap

plt.figure(figsize=(10, 5))

total = data['Population'].sum()
left = 0

for i in range(len(data)):

    width = data['Population'][i] / total

    rectangle = Rectangle(
        (left, 0),
        width,
        1
    )

    plt.gca().add_patch(rectangle)

    plt.text(
        left + width / 2,
        0.5,
        data['City'][i],
        ha='center',
        va='center'
    )

    left = left + width

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

plt.title("City-wise Population Treemap")

plt.show()