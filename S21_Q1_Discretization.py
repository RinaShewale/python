# S21_Q1_Discretization.py


import pandas as pd

book_records = pd.DataFrame({
    'Book_ID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Book_Price': [100, 150, 200, 250, 300, 350, 400, 450]
})

print("Original Data:")
print(book_records)

book_records['Price_Category'] = pd.cut(
    book_records['Book_Price'],
    bins=4,
    labels=['Low', 'Medium', 'High', 'Very High']
)

print("\nDiscretized Data:")
print(book_records)







# S21_Q2_WorldCities.py


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
plt.title("World Cities Population Visualization")
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