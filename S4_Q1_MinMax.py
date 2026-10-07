import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Product_ID": [101, 102, 103, 104, 105],
    "Price": [100, 200, 150, 300, 250],
    "Quantity_Sold": [10, 20, 15, 25, 30],
    "Discount": [5, 10, 15, 20, 25],
    "Revenue": [950, 3600, 1912, 6000, 5625]
}

df = pd.DataFrame(data)

scaler = MinMaxScaler()

columns = ["Price", "Quantity_Sold", "Discount", "Revenue"]

df[columns] = scaler.fit_transform(df[columns])

print(df)




# S4_Q2_Books.py


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud

data = {
    "Book Title": [
        "Python Basics", "Data Science", "Machine Learning",
        "Web Development", "Artificial Intelligence", "Python Advanced"
    ],
    "Author": [
        "John", "Smith", "John", "David", "Smith", "David"
    ],
    "Publication Year": [2010, 2012, 2015, 2018, 2020, 2022],
    "Average Rating": [4.1, 4.3, 4.5, 4.0, 4.7, 4.6],
    "Number of Ratings": [100, 150, 200, 120, 250, 180]
}

df = pd.DataFrame(data)

period1 = set(df[df["Publication Year"] <= 2015]["Author"])
period2 = set(df[df["Publication Year"] > 2015]["Author"])

venn2([period1, period2], set_labels=("2010-2015", "2016-2022"))
plt.title("Authors in Two Publication Periods")
plt.show()

text = " ".join(df["Book Title"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Book Titles Word Cloud")
plt.show()