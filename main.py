import colored as colored
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Zaczytaj dane
# 2. Sprawdź wymiary
# 3. Usuń duplikaty
# 4. Zmień typ danych
# 5. Uzupełnij/ usuń braki
# 6. Zmień dane kategoryczne na numeryczne
# 7. Znormalizuj dane
# 8. Podziel zbiór na uczący i testowy

# Wczytywanie
# 1. zaczytaj dane z pliku csv
df = pd.read_csv("train.csv", sep=",", encoding='utf-8')
# 2. sprawdź liczbę kolumn i wierszy (columns, entries)
print("Info:")
df.info()
# wyświetl część tabeli
print("Nagłówek (początkowy):")
print(df.head())
# 3. usuń wiersze z duplikatami id
df.drop_duplicates(subset="ID", inplace=True)

# 4. zamiana danych
FeaturesToConvert = ['Age', 'Annual_Income',
                     'Num_of_Loan', 'Num_of_Delayed_Payment',
                     'Changed_Credit_Limit', 'Outstanding_Debt',
                     'Amount_invested_monthly', 'Monthly_Balance']
# Usuwanie błędów
for feature in FeaturesToConvert:
    uniques = df[feature].unique()

    # Usuwanie zbędnych znaków
    df[feature] = df[feature].str.strip('-_')

    # NaN w pustych kolumnach
    df[feature] = df[feature].replace({'': np.nan})

    # Typ danych
    df[feature] = df[feature].astype('float64')

# Uzupełnienie braku śedniej
df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(method='pad')

# Dane kategoryczne na numeryczne
# oprócz occupation trzeba też: credit_mix, payment_of_min_amount, payment_bhvr
# Pojedynczo
le = LabelEncoder()
df.Occupation = le.fit_transform(df.Occupation)
print(df.Occupation)  # widzę liczby - działa
# Albo razem
columns = ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Type_of_Loan', 'Credit_Score']
df[columns] = df[columns].apply(le.fit_transform)
print(df.Payment_Behaviour)  # też działa

# Zamiana dat
# print(df['Credit_History_Age'].head)
# ohe = OneHotEncoder(handle_unknown='ignore')
# ohe.fit(df.Credit_History_Age)

# Eksport
df.to_csv('convertedtrain.csv')

# Standaryzacja/normalizacja danych
scaler = MinMaxScaler()
col_float = ['Age', 'Annual_Income',
'Delay_from_due_date', 'Num_of_Delayed_Payment',
'Outstanding_Debt',
'Total_EMI_per_month', 'Monthly_Balance']

for i in df[col_float]:
    df[i] = scaler.fit_transform(df[[i]])

# Eksport
df.to_csv('normalizedtrain.csv')

