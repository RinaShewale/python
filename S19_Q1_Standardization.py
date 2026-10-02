# S19_Q1_Standardization.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'Age': [22, 25, 28, 30, 35],
    'Salary': [25000, 30000, 40000, 50000, 60000],
    'Experience': [1, 2, 4, 6, 10],
    'Working_Hours': [8, 8, 9, 9, 10],
    'Performance_Score': [60, 65, 72, 80, 90]
})

print("Original Data:")
print(data)

scaler = StandardScaler()

transformed_data = scaler.fit_transform(data)

transformed_data = pd.DataFrame(
    transformed_data,
    columns=data.columns
)

print("\nStandardized Data:")
print(transformed_data)










# S19_Q2_Venn_WordCloud.py


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud

data = pd.DataFrame({
    'Book Title': [
        'Python Basics',
        'Data Science Guide',
        'Machine Learning',
        'Web Development',
        'Python Projects',
        'Data Analytics',
        'Artificial Intelligence',
        'Web Design'
    ],
    'Author': [
        'John',
        'Smith',
        'John',
        'David',
        'Smith',
        'David',
        'John',
        'Smith'
    ],
    'Publication Year': [2010, 2012, 2015, 2017, 2018, 2020, 2021, 2023],
    'Average Rating': [4.2, 4.5, 4.6, 4.1, 4.7, 4.3, 4.8, 4.4],
    'Number of Ratings': [100, 150, 200, 120, 180, 160, 250, 140]
})

print("Book Data:")
print(data)

# Venn Diagram

period1 = set(
    data[data['Publication Year'] <= 2016]['Author']
)

period2 = set(
    data[data['Publication Year'] > 2016]['Author']
)

venn2(
    [period1, period2],
    set_labels=('2010-2016', '2017-2023')
)

plt.title("Authors in Different Time Periods")
plt.show()


# Word Cloud

titles = ' '.join(data['Book Title'])

wordcloud = WordCloud(
    width=800,
    height=400
).generate(titles)

plt.imshow(wordcloud)
plt.axis("off")
plt.title("Book Titles Word Cloud")
plt.show()