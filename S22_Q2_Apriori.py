# S22_Q2_Apriori.py

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = [
    ['Basmati Rice', 'Toor Dal', 'Turmeric Powder', 'Cooking Oil'],
    ['Atta', 'Sugar', 'Tea', 'Milk'],
    ['Rice', 'Moong Dal', 'Salt', 'Cooking Oil'],
    ['Wheat Flour', 'Chickpeas', 'Turmeric Powder', 'Cumin Seeds'],
    ['Poha', 'Peanuts', 'Onions', 'Cooking Oil'],
    ['Basmati Rice', 'Paneer', 'Garam Masala', 'Tomatoes'],
    ['Atta', 'Potatoes', 'Onions', 'Cooking Oil']
]

print("Number of Transactions:", len(data))
print("\nTransactions:")
for i, transaction in enumerate(data, 1):
    print(i, transaction)

# Convert transactions into numeric format

encoder = TransactionEncoder()
encoded_data = encoder.fit(data).transform(data)

df = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

print("\nEncoded Data:")
print(df)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Apriori - Minimum Support 0.3

frequent_03 = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets - Support 0.3:")
print(frequent_03)

rules_03 = association_rules(
    frequent_03,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules - Support 0.3:")
print(rules_03)


# Apriori - Minimum Support 0.4

frequent_04 = apriori(
    df,
    min_support=0.4,
    use_colnames=True
)

print("\nFrequent Itemsets - Support 0.4:")
print(frequent_04)

rules_04 = association_rules(
    frequent_04,
    metric="confidence",
    min_threshold=0.5
)

print("\nAssociation Rules - Support 0.4:")
print(rules_04)