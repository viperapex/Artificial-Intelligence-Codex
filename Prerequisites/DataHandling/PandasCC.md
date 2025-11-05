import pandas as pd

df = pd.read_csv('salaries.csv')
myfilter = df['Salary'] > 60

print(df[df['Salary'] > 60000])
