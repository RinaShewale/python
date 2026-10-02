# S12_Q1_Titanic.py

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("titanic.csv")

print(data.head())

plt.hist(data['Pclass'], bins=3)
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Passengers in Each Class")
plt.show()

survival = data['Survived'].value_counts()

plt.pie(
    survival,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%"
)
plt.title("Titanic Survival Distribution")
plt.show()





# S12_Q2_PolynomialRegression.py


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.DataFrame({
    'Danceability': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'Energy': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
    'Tempo': [80, 90, 100, 110, 120, 130, 140, 150],
    'Popularity': [30, 35, 42, 50, 58, 68, 78, 85]
})

X = data[['Danceability', 'Energy', 'Tempo']]
y = data['Popularity']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.scatter(data['Danceability'], y)

x = data[['Danceability']].sort_values('Danceability')
x_poly = poly.transform(
    pd.DataFrame({
        'Danceability': x['Danceability'],
        'Energy': data.loc[x.index, 'Energy'],
        'Tempo': data.loc[x.index, 'Tempo']
    })
)

plt.plot(x['Danceability'], model.predict(x_poly))
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.title("Polynomial Regression")
plt.show()