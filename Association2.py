# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:36:12 2022

@author: Gopinath
"""

# -*- coding: utf-8 -*-
"""
Apriori
Importing the libraries
"""

pip install apyori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
df = pd.read_csv("my_movies.csv")
df
df.shape
df.info()
df1=df.iloc[:,5:]
df1
#Appriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
# With 10% Support
frequent_itemsets=apriori(df1,min_support=0.1,use_colnames=True)
frequent_itemsets
# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules
rules.sort_values('lift',ascending=False)

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]

# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# With 20% Support
frequent_itemsets2=apriori(df1,min_support=0.20,use_colnames=True)
frequent_itemsets2
# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2
# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# With 5% Support
frequent_itemsets3=apriori(df1,min_support=0.05,use_colnames=True)
frequent_itemsets3

# With 90% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.9)
rules3

rules3[rules3.lift>1]
# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()




















