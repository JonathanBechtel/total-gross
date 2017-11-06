from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LassoCV

url = "https://raw.githubusercontent.com/JonathanBechtel/total-gross/master/gpr2.csv"
data = pd.read_csv(url)

del data['Deal']
data.rename(index=str, columns={"textbox4": "Total_Gross"}, inplace=True)

data['Total_Gross'] = data['Total_Gross'].str.translate(None, " $,)").str.replace("(", "-")
data['Total_Gross'] = data['Total_Gross'].astype(int)

data = data[(data.Year > 2007) & (data.Year < 2018)]

total_gross = data['Total_Gross']
length      = len(total_gross)

fifth_percentile           = total_gross.nlargest(int(round(length*0.05))).iloc[-1]
ninety_fifth_percentile    = total_gross.nsmallest(int(round(length*0.05))).iloc[-1]

top_5               = total_gross[total_gross > fifth_percentile]
bottom_5            = total_gross[total_gross < ninety_fifth_percentile]
five_percent_tails  = pd.concat([top_5, bottom_5], axis=0)
data                = data[~data.Total_Gross.isin(five_percent_tails).values]

plt.figure(1)
plt.hist(data['Total_Gross'], bins=10)
plt.title("Distribution of Middle 90% of Customers")
plt.ylabel("Frequency")
plt.show()

data['Lender']           = data['Lender'].fillna("Cash")
data['Marketing_Source'] = data['Marketing_Source'].fillna("None")
data['Year']             = data['Year'].fillna(2015)

data = pd.get_dummies(data)
y    = data['Total_Gross']
del data['Total_Gross']
X = data

lasso = LassoCV(cv=5)
lasso.fit(X, y)

coeff = pd.DataFrame({'Features': data.columns, 'Coefficients': lasso.coef_})
coeff = coeff.sort_values(by='Coefficients', ascending=False)