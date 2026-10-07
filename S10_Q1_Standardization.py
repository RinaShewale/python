import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [22, 25, 30, 35, 40],
    "Salary": [25000, 30000, 40000, 50000, 60000],
    "Experience": [1, 3, 5, 8, 12],
    "Working Hours": [7, 8, 8, 9, 10],
    "Performance_Score": [60, 70, 75, 85, 90]
}

df = pd.DataFrame(data)

scaler = StandardScaler()

transformed_data = scaler.fit_transform(df)

result = pd.DataFrame(transformed_data, columns=df.columns)

print("Original Data:")
print(df)

print("\nStandardized Data:")
print(result)







# S9_Q2_Books.py


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud

data = {
    "Book Title": [
        "Python Basics",
        "Data Science",
        "Machine Learning",
        "Web Development",
        "Artificial Intelligence",
        "Python Advanced"
    ],
    "Author": [
        "John",
        "Smith",
        "John",
        "David",
        "Smith",
        "David"
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

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Book Titles Word Cloud")
plt.show()