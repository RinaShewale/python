import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    "Product_name": ["A", "B", "C", "D", "E"],
    "Price": [100, 120, 500, 550, 300],
    "Rating": [4.1, 4.2, 4.8, 4.7, 4.3],
    "Number_of_Sales": [100, 120, 500, 550, 250]
}

df = pd.DataFrame(data)

X = df[["Price", "Rating", "Number_of_Sales"]]

Z = linkage(X, method="ward")

dendrogram(Z, labels=df["Product_name"].values)

plt.title("Dendrogram of Similar Products")
plt.xlabel("Products")
plt.ylabel("Distance")

plt.show()




# S10_Q2_Transformation.py


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

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
normal = Normalizer()

minmax_data = minmax.fit_transform(df)
standard_data = standard.fit_transform(df)
normal_data = normal.fit_transform(df)

print("Min-Max Scaling:")
print(pd.DataFrame(minmax_data, columns=df.columns))

print("\nStandardization:")
print(pd.DataFrame(standard_data, columns=df.columns))

print("\nNormalization:")
print(pd.DataFrame(normal_data, columns=df.columns))