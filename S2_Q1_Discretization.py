# S2_Q1_Discretization.py

import pandas as pd

age_records = pd.DataFrame({
    'Person_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Age': [12, 18, 25, 32, 40, 48, 55, 65]
})

age_records['Age_Group'] = pd.cut(
    age_records['Age'],
    bins=4,
    labels=['Young', 'Adult', 'Middle Age', 'Senior']
)

print("Original Data:")
print(age_records[['Person_ID', 'Age']])

print("\nDiscretized Data:")
print(age_records)



# S2_Q2_LinearRegression.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    'Likes': [100, 200, 300, 400, 500, 600, 700, 800],
    'Views': [1000, 1800, 2500, 3300, 4200, 4900, 5700, 6500]
})

X = data[['Likes']]
y = data['Views']

model = LinearRegression()
model.fit(X, y)

predicted = model.predict(X)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(X, y)
plt.plot(X, predicted)
plt.xlabel("Likes")
plt.ylabel("Views")
plt.title("YouTube Views Prediction")
plt.show()