# S11_Q1_BarChart.py

import matplotlib.pyplot as plt

departments = ["HR", "Finance", "IT", "Sales", "Marketing"]
employees = [20, 30, 50, 40, 25]

plt.bar(departments, employees)

plt.xlabel("Departments")
plt.ylabel("Number of Employees")
plt.title("Employee Distribution Among Departments")

plt.show()




# S11_Q2_KMeans_Ecommerce.py


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Ecommerce_Customers.csv")

print("Original Data:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

data = data.dropna()

features = data[
    ['Avg_Session_Length', 'Time_on_App',
     'Time_on_Website', 'Length_of_Membership']
]

scaler = StandardScaler()
X = scaler.fit_transform(features)

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

data['Cluster'] = model.labels_

print("\nClustered Data:")
print(data[['Avg_Session_Length',
            'Time_on_App',
            'Time_on_Website',
            'Length_of_Membership',
            'Cluster']])

plt.scatter(X[:, 0], X[:, 1], c=data['Cluster'])
plt.xlabel("Average Session Length")
plt.ylabel("Time on App")
plt.title("E-commerce Customer Clusters")
plt.show()