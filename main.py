import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# zaczytanie danych z pliku
df_train = pd.read_csv("train.csv", sep=",", encoding='utf-8')
# sprawdzenie rozmiaru
print(df_train.shape)
# sprawdzenie nazw kolumn i ich typów
print(df_train.info())
# wyświetl część tabeli
df_train.head()
# usunięcie wierszy z duplikatami id
df_train.drop_duplicates(subset="ID", inplace=True)

# nazwy kolumn dla danych docelowo numerycznych
FeaturesToConvert = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                     'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
                     'Monthly_Balance']

# nazwy kolumn dla danych kategorycznych

# liczności danych kategorii
sns.countplot(df_train['Credit_Score'])

# sprawdzenie danych
for i in df_train.columns:
    print(df_train[i].value_counts())
    print('*' * 50)

# wyświetl liczbę pustych wartosci
print(df_train.isnull().sum().sort_values(ascending=False))

########## dane numeryczne #########################
# usuń zbędne znaki '-’ , '_'
for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].str.strip('-_')
# puste kolumny zastąp NAN
for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].replace({'': np.nan})
# zmien typ zmiennych ilościowych
for feature in FeaturesToConvert:
    df_train[feature] = df_train[feature].astype('float64')

# uzupełnij braki średnią
df_train['Monthly_Inhand_Salary'] = df_train['Monthly_Inhand_Salary'].fillna(method='pad')
df_train['Monthly_Balance'] = df_train['Monthly_Inhand_Salary'].fillna(method='pad')
df_train['Type_of_Loan'] = df_train['Type_of_Loan'].fillna(method='ffill')
df_train['Credit_History_Age'] = df_train['Type_of_Loan'].fillna(method='pad')
df_train['Num_of_Delayed_Payment'] = df_train['Num_of_Delayed_Payment'].fillna(method='pad')
df_train['Amount_invested_monthly'] = df_train['Amount_invested_monthly'].fillna(method='pad')
df_train['Changed_Credit_Limit'] = df_train['Changed_Credit_Limit'].fillna(method='pad')
df_train['Num_Credit_Inquiries'] = df_train['Num_Credit_Inquiries'].fillna(method='pad')

print(df_train.isnull().sum().sort_values(ascending=False))

# zastąpienie nierealnych wartości medianą
for i in df_train.Age.values:
    if (i > 118 or i < 0):
        df_train.Age.replace(i, np.median(df_train.Age), inplace=True)

############ zmienne kategoryczne #####################
# stwórz obiekt enkodera
le = LabelEncoder()

CatFeatures = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount',
               'Payment_Behaviour', 'Credit_Score']

# zakoduj etykiety słowne numerycznymi

df_train[CatFeatures] = df_train[CatFeatures].apply(LabelEncoder().fit_transform)

# usuń nieistotne kolumny
irrelevant = ['ID', 'Customer_ID', 'Name', 'SSN']
df_train = df_train.drop(irrelevant, axis=1)

#######################################################
# sprawdź transformacje
print(df_train.shape)
print('\n**********************************\n')
print(df_train.info())
print('\n**********************************\n')
print(df_train.describe().transpose())
print('\n**********************************\n')
######################################################

