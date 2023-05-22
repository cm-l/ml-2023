import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("Salary Data.csv", sep = "," , encoding= 'utf-8')
# biekt enkodera
le = LabelEncoder()
CatFeatures = ['Gender','Education Level','Job Title']

# zakoduj etykiety słowne numerycznymi
df[CatFeatures] = df[CatFeatures].apply(LabelEncoder().fit_transform)



X=df.loc[:,['Age','Gender','Education Level','Job Title','Years of Experience']]
Y=df.loc[:,['Salary']]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=100)

regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))



# Wersja Regresja liniowa LASSO
lasso = linear_model.Lasso(alpha=1.0)
lasso.fit(X_train, Y_train) # dane treningowe

y_predicted_by_lasso = lasso.predict(X_test)

# The coefficients
print("Coefficients of LASSO regr: \n", lasso.coef_)

# Wersja Wielomianowa
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_x = poly.fit_transform(X_train)
poly_x_test = poly.fit_transform(X_test)

poly_model = linear_model.LinearRegression()
# Trained
poly_model.fit(poly_x, Y_train)
# Prediced
Y_predicted_by_poly = poly_model.predict(poly_x_test)
# The coefficients (nie do zinterpretowania)
print("Coefficients of poly regr: \n", poly_model.coef_)

# Wersja KNN (k = 3)
#skalowanie
scale = MinMaxScaler(feature_range=(0,1))
X_train_scaling = scale.fit_transform(X_train)
X_test_scaling = scale.fit_transform(X_test)

neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train_scaling, Y_train)
neigh.predict(X_test_scaling)
# R^2 dla knn
print("Coefficient of determination for KNN: \n", neigh.score(X_test_scaling, Y_test))

