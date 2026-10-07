import pandas as pd

data = {
    "Book_ID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Book_Price": [100, 150, 200, 250, 300, 350, 400, 450]
}

book_records = pd.DataFrame(data)

book_records["Price_Group"] = pd.cut(
    book_records["Book_Price"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

print("Original Data:")
print(book_records[["Book_ID", "Book_Price"]])

print("\nDiscretized Data:")
print(book_records)




# S11_Q2_World_Cities.py


import pandas as pd
import matplotlib.pyplot as plt
import squarify

data = {
    "city": ["Mumbai", "London", "New York", "Tokyo", "Delhi"],
    "country": ["India", "UK", "USA", "Japan", "India"],
    "latitude": [19.07, 51.50, 40.71, 35.68, 28.61],
    "longitude": [72.87, -0.12, -74.00, 139.69, 77.20],
    "population": [20, 9, 8, 14, 19]
}

df = pd.DataFrame(data)

plt.scatter(
    df["longitude"],
    df["latitude"],
    s=df["population"] * 100,
    c=df["population"],
    cmap="viridis"
)

for i in range(len(df)):
    plt.text(df["longitude"][i], df["latitude"][i], df["city"][i])

plt.title("World Cities Population")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Population")

plt.show()

plt.figure(figsize=(8, 5))

squarify.plot(
    sizes=df["population"],
    label=df["city"],
    color=["red", "blue", "green", "orange", "purple"]
)

plt.title("Population Distribution of Cities")
plt.axis("off")

plt.show()