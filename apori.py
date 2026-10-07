import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Library", "Seminar"],
    ["Workshop", "Project"],
    ["Library", "Assignment"],
    ["Seminar", "Workshop"],
    ["Assignment", "Seminar"],
    ["Workshop", "Assignment"],
    ["Project", "Library"],
    ["Library", "Seminar", "Project"],
    ["Library", "Seminar", "Workshop"]
]

te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(data, columns=te.columns_)

for support in [0.2, 0.3]:
    print("\nMinimum Support:", support)

    frequent_itemsets = apriori(
        df,
        min_support=support,
        use_colnames=True
    )

    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    rules = association_rules(
        frequent_itemsets,
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