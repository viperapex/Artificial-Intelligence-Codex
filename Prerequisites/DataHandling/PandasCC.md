[[Pandas-Numpy]]

## Standard Import Syntax

```python
import pandas as pd
```

## Reading CSV 

```python
df = pd.read_csv('csv.csv')

print(df)
'''
Output:
     Name  Salary  Age
0    John   50000   34
1   Sally  120000   45
2  Alyssa   80000   27
'''
```

## Describe Dataframe

- **Count**: Number of non-null values
- **Mean**: Average value
- **Std**: Standard deviation (measure of spread)
- **Min**/**Max**: Minimum and maximum values
- **Percentiles**: 25th (Q1), 50th (median), and 75th (Q3) quartiles

For **categorical/object columns**, it returns:

- **Count**, **Unique**, **Top** (most frequent category), and **Freq** (frequency of the top category).

```python
df.describe()
#df.describe(include='all')  # Includes categorical columns
#df.describe(percentiles=[0.1, 0.9])  # Custom percentiles
'''
Output:
              Salary        Age
count       3.000000   3.000000
mean    83333.333333  35.333333
std     35118.845843   9.073772
min     50000.000000  27.000000
25%     65000.000000  30.500000
50%     80000.000000  34.000000
75%    100000.000000  39.500000
max    120000.000000  45.000000
'''
```
## Accessing Columns and Rows

### Specific column

```python
df['Salary']
'''
Output:
0     50000
1    120000
2     80000
Name: Salary, dtype: int64
'''
```

### Multiple columns

```python
df['Salary','Name']
'''
Output:
   Salary    Name
0   50000    John
1  120000   Sally
2   80000  Alyssa
'''
```

### Max value in column

```python
df['Salary'].max()
#Output: 120000
```

## Boolean Filters/Masking

```python
myfilter = df['Salary'] > 60
'''
Output:
0    True
1    True
2    True
'''
```

```python
df[df['Salary'] > 60000]
'''
Output:
     Name  Salary  Age
1   Sally  120000   45
2  Alyssa   80000   27
'''
```



![[Pandas_Cheat_Sheet.pdf]]