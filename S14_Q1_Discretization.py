# S14_Q1_Discretization.py

import pandas as pd

age_records = pd.DataFrame({
    'Person_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Age': [12, 18, 25, 32, 40, 48, 55, 65]
})

print("Original Data:")
print(age_records)

age_records['Age_Group'] = pd.qcut(
    age_records['Age'],
    q=4,
    labels=['Young', 'Adult', 'Middle Age', 'Senior']
)

print("\nDiscretized Data:")
print(age_records)







# S14_Q2_Treemap_3DScatter.py

# pip install squarify

import pandas as pd
import matplotlib.pyplot as plt
import squarify

data = pd.DataFrame({
    'City': ['Pune', 'Mumbai', 'Nashik', 'Nagpur', 'Delhi', 'Jaipur'],
    'Area': [331, 603, 259, 227, 1484, 484],
    'Temperature': [25, 28, 24, 27, 30, 29],
    'Rainfall': [722, 2160, 700, 1100, 800, 600],
    'Population_Density': [7500, 20000, 5000, 4500, 11500, 6500]
})

print(data)

plt.figure(figsize=(8, 5))

squarify.plot(
    sizes=data['Population_Density'],
    label=data['City'],
    color=['red', 'blue', 'green', 'orange', 'purple', 'cyan']
)

plt.title("Population Density of Cities")
plt.axis("off")
plt.show()

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    data['Area'],
    data['Temperature'],
    data['Rainfall'],
    c=data['Population_Density'],
    cmap='viridis',
    s=100
)

ax.set_xlabel("Area (sq. km)")
ax.set_ylabel("Average Temperature (°C)")
ax.set_zlabel("Rainfall (mm)")
ax.set_title("3D Scatter Plot of Cities")

fig.colorbar(scatter, label="Population Density")

plt.show()