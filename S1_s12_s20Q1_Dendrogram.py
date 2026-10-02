# S1_Q1_Dendrogram.py

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Product data
data = {
    'Price': [100, 120, 500, 520, 110, 490],
    'Rating': [4.2, 4.0, 4.8, 4.7, 4.1, 4.9],
    'Sales': [100, 120, 500, 520, 110, 480]
}

df = pd.DataFrame(data)

print("Product Data:")
print(df)

# Create clusters
linked = linkage(df, method='ward')

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linked)

plt.title("Product Dendrogram")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.show()







# S1_Q2_KMeans_Iris.py

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


iris = load_iris()
X = iris.data[:, :2]


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)


labels = kmeans.labels_
centers = kmeans.cluster_centers_


plt.scatter(X[:, 0], X[:, 1], c=labels)


plt.scatter(centers[:, 0], centers[:, 1],
            marker='X', s=200)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris K-Means Clustering")

plt.show()









# S13_Q2_PolynomialRegression.py


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("car data.csv")

X = data[['Vehicle_Age', 'Present_Price', 'Kms_Driven']]
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
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

