import matplotlib.pyplot as plt

courses = ["BCA", "BBA", "B.Com", "B.Sc", "BA"]
students = [60, 45, 75, 50, 35]

plt.bar(courses, students, color=["red", "blue", "green", "orange", "purple"])

plt.title("Number of Students Enrolled in Each Course")
plt.xlabel("Courses")
plt.ylabel("Number of Students")

plt.show()



# S2_Q2_Student_KNN.py
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Basmati Rice", "Toor Dal", "Turmeric Powder", "Cooking Oil"],
    ["Atta", "Sugar", "Tea", "Milk"],
    ["Rice", "Moong Dal", "Salt", "Cooking Oil"],
    ["Wheat Flour", "Chickpeas", "Turmeric Powder", "Cumin Seeds"],
    ["Poha", "Peanuts", "Onions", "Cooking Oil"],
    ["Basmati Rice", "Paneer", "Garam Masala", "Tomatoes"],
    ["Atta", "Potatoes", "Onions", "Cooking Oil"]
]

print("Transactions:")
print(transactions)

te = TransactionEncoder()
data = te.fit_transform(transactions)

df = pd.DataFrame(data, columns=te.columns_)

df = df.fillna(False)

for support in [0.3, 0.4]:

    print("\nMinimum Support:", support)

    items = apriori(df, min_support=support, use_colnames=True)

    print("\nFrequent Itemsets:")
    print(items)

    rules = association_rules(
        items,
        metric="confidence",
        min_threshold=0.5
    )

    print("\nAssociation Rules:")
    print(rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]])