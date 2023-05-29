import csv

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Import baskets
baskets = []
with open('GroceryStoreDataSet.csv', 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        basket = line[0].split(',')
        baskets.append(basket)

print("Baskets as a list of lists:")
print(baskets)

# Encode baskets
encoder = TransactionEncoder()
encoded_data = encoder.fit_transform(baskets) # this is very hard to read

# Convert to df so it is readable
df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)

print(df_encoded.head())

# Find frequent itemsets using Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True, verbose=1)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Print the association rules
print("Found rules:")
print(rules)

# Find the pair of products with the highest confidence
highest_confidence_rule = rules[rules['confidence'] == rules['confidence'].max()]
print("Pair with highest confidence:")
print(highest_confidence_rule[['antecedents', 'consequents', 'confidence']])

# Find the pair of products with the highest lift
highest_lift_rule = rules[rules['lift'] == rules['lift'].max()]
print("Pair with highest lift:")
print(highest_lift_rule[['antecedents', 'consequents', 'lift']])