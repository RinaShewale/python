
# S4_Q1_PieChart.py
import matplotlib.pyplot as plt

payment_methods = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
customers = [150, 80, 60, 40, 30]

plt.pie(customers, labels=payment_methods, autopct='%1.1f%%')
plt.title("Preferred Payment Methods")
plt.show()






# S4_Q2_KMeans_Ecommerce.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Ecommerce_Customers.csv")

print("Original Data:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

data = data.dropna()

features = data[['Avg_Session_Length', 'Time_on_App', 'Time_on_Website', 'Length_of_Membership']]

scaler = StandardScaler()
X = scaler.fit_transform(features)

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

data['Cluster'] = model.labels_

print("\nClustered Data:")
print(data[['Avg_Session_Length', 'Time_on_App', 'Time_on_Website', 'Length_of_Membership', 'Cluster']])

print("\nCluster Centers:")
print(model.cluster_centers_)