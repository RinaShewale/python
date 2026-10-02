# S5_Q1_LabelEncoding.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

customer_DataFrame = pd.DataFrame({
    'Customer_ID': [101, 102, 103, 104, 105],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'City': ['Pune', 'Mumbai', 'Nashik', 'Pune', 'Mumbai'],
    'Membership_Type': ['Gold', 'Silver', 'Gold', 'Bronze', 'Silver']
})

print("Before Encoding:")
print(customer_DataFrame)

encoder = LabelEncoder()

customer_DataFrame['Gender'] = encoder.fit_transform(customer_DataFrame['Gender'])
customer_DataFrame['City'] = encoder.fit_transform(customer_DataFrame['City'])
customer_DataFrame['Membership_Type'] = encoder.fit_transform(customer_DataFrame['Membership_Type'])

print("\nAfter Encoding:")
print(customer_DataFrame)




# S5_Q2_Netflix_Amazon.py

# pip install matplotlib-venn

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

customer_data = pd.DataFrame({
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Netflix': [1, 1, 1, 0, 0, 0, 1, 0],
    'Amazon_Prime': [1, 0, 1, 1, 0, 0, 0, 1]
})

netflix = set(customer_data[customer_data['Netflix'] == 1]['Customer_ID'])
amazon = set(customer_data[customer_data['Amazon_Prime'] == 1]['Customer_ID'])

only_netflix = len(netflix - amazon)
only_amazon = len(amazon - netflix)
both = len(netflix & amazon)

print("Only Netflix:", only_netflix)
print("Only Amazon Prime:", only_amazon)
print("Both:", both)

venn2([netflix, amazon], set_labels=('Netflix', 'Amazon Prime'))
plt.title("Netflix and Amazon Prime Customers")
plt.show()

labels = ['Only Netflix', 'Only Amazon Prime', 'Both']
values = [only_netflix, only_amazon, both]

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Customer Subscription Distribution")
plt.show()