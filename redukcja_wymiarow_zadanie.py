import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

# logistic reg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 2. plik z danymi cleaned_data.csv
df = pd.read_csv("cleaned_data.csv", sep=",", encoding='utf-8')

print(df) # dane początkowe
print(df.columns) # cechy

exclude_filter = ~df.columns.isin(['Unnamed: 0', 'Credit_Score'])

pca = PCA().fit(df.loc[:, exclude_filter])
plt.plot(np.cumsum(pca.explained_variance_ratio_)) # podkreslenie ______

plt.xlabel('Components:')
plt.ylabel('Explained Variance Cumulative')
plt.gcf().set_size_inches(10, 6) # resize

# plt.show() # pycharm

pca = PCA(svd_solver='full', n_components=0.95)
principal_components = pca.fit_transform(df.loc[:, exclude_filter])
principal_df = pd.DataFrame(data=principal_components)
print(principal_df.head())

X_train, X_test, Y_train, Y_test = train_test_split(principal_df, df['Credit_Score'], test_size=0.25, random_state=77)


# Regresja logistyczna
# 1
logreg = LogisticRegression(random_state=77)
# 2
logreg.fit(X=X_train, y=Y_train)
# 3
predicted = logreg.predict(X_test)

# Macierz pomyłek
mp = confusion_matrix(Y_test, predicted)
mp_plot = ConfusionMatrixDisplay(mp, display_labels=logreg.classes_).plot() # wyswietl
mp_plot.plot()
plt.show()

cr = classification_report(Y_test, predicted) # raport

print(cr)

