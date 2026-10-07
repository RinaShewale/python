import pandas as pd

data = {
    "Person_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [18, 22, 25, 28, 31, 35, 40, 45, 50, 60]
}

age_records = pd.DataFrame(data)

age_records["Age_Group"] = pd.qcut(
    age_records["Age"],
    q=3,
    labels=["Young", "Middle", "Old"]
)

print("Original Data:")
print(age_records[["Person_ID", "Age"]])

print("\nDiscretized Data:")
print(age_records)




# S7_Q2_City_Visualization.py


import pandas as pd
import matplotlib.pyplot as plt
import squarify
from mpl_toolkits.mplot3d import Axes3D

data = {
    "City": ["Mumbai", "Pune", "Nashik", "Nagpur", "Delhi"],
    "Area": [603, 516, 259, 227, 1484],
    "Average Temperature": [27, 25, 24, 28, 26],
    "Rainfall": [2422, 722, 700, 1100, 800],
    "Population Density": [21000, 7000, 6000, 12000, 11500]
}

df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))

squarify.plot(
    sizes=df["Population Density"],
    label=df["City"],
    color=["red", "blue", "green", "orange", "purple"]
)

plt.title("Population Density of Cities")
plt.axis("off")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["Area"],
    df["Average Temperature"],
    df["Rainfall"],
    c=df["Population Density"],
    cmap="viridis",
    s=100
)

ax.set_xlabel("Area (sq. km)")
ax.set_ylabel("Average Temperature (°C)")
ax.set_zlabel("Rainfall (mm)")
ax.set_title("3D Scatter Plot of Cities")

plt.show()