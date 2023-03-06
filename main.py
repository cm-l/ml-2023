import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
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
# 2. sprawdź liczbę kolumn i
print("Info:")
df.info()
# wyświetl część tabeli
print("Nagłówek:")
df.head()
# 3. usuń wiersze z duplikatami id
df.drop_duplicates(subset="ID", inplace=True)

# 4.

