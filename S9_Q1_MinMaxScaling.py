# S9_Q1_MinMaxScaling.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame({
    'Product_ID': [101, 102, 103, 104, 105],
    'Price': [100, 200, 300, 400, 500],
    'Quantity_Sold': [10, 20, 15, 30, 25],
    'Discount': [5, 10, 15, 20, 25],
    'Revenue': [950, 3600, 3825, 9600, 9375]
})

print("Original Data:")
print(data)

scaler = MinMaxScaler()

data[['Price', 'Quantity_Sold', 'Discount', 'Revenue']] = scaler.fit_transform(
    data[['Price', 'Quantity_Sold', 'Discount', 'Revenue']]
)

print("\nTransformed Data:")
print(data)







# S9_Q2_BookVennWordCloud.py
S9_Q2_BookVennWordCloud.py


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud

data = pd.DataFrame({
    'Book Title': [
        'Python Basics', 'Data Science Guide', 'Machine Learning',
        'Web Development', 'Python Projects', 'Data Analytics',
        'Artificial Intelligence', 'Web Design'
    ],
    'Author': [
        'John', 'Smith', 'John', 'David',
        'Smith', 'David', 'John', 'Smith'
    ],
    'Publication Year': [2010, 2012, 2015, 2017, 2018, 2020, 2021, 2023],
    'Average Rating': [4.2, 4.5, 4.6, 4.1, 4.7, 4.3, 4.8, 4.4],
    'Number of Ratings': [100, 150, 200, 120, 180, 160, 250, 140]
})

print("Book Data:")
print(data)

period1 = set(data[data['Publication Year'] <= 2016]['Author'])
period2 = set(data[data['Publication Year'] > 2016]['Author'])

print("\nAuthors in First Period:", period1)
print("Authors in Second Period:", period2)

venn2([period1, period2], set_labels=('2010-2016', '2017-2023'))
plt.title("Authors in Different Time Periods")
plt.show()

titles = ' '.join(data['Book Title'])

wordcloud = WordCloud(width=800, height=400).generate(titles)

plt.imshow(wordcloud)
plt.axis("off")
plt.title("Book Titles Word Cloud")
plt.show()