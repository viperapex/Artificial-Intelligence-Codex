# Dataviz with Matplotlib

## Standard Import Syntax
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


## Plotting a line

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline (only for jupyter nb, helps avoid using plt.show() after every plot code for rendering graphs)

x = np.arange(0, 10)
print(x)

y = x**2

plt.plot(x, y, 'r--') # red dasshed line
plt.xlim(0, 4) # x limit
plt.ylim(0, 10) # y limit
plt.title("Title Here")
plt.xlabel('X Label here')
plt.ylabel('Y Label here')
plt.show()plt.show()
```

![Figure 1](imgs/Figure_1.png)


## Imshow plot

```python
mat = np.arange(0, 100).reshape(10, 10)
print(mat)
'''
Output:
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]
'''

plt.imshow(mat, cmap='coolwarm')

plt.show()
```

![[Figure_2 1.png]]

```python
mat = np.randint(0,1000,(10,10))
plt.imshow(mat)
```

![[Figure_3 1.png]]

## Pandas Visualization

```python
df = pd.read_csv('salaries.csv')
print(df)
'''
Output
     Name  Salary  Age
0    John   50000   34
1   Sally  120000   45
2  Alyssa   80000   27
'''

df.plot(x='Salary', y='Age', kind='scatter')

plt.show()
```

![[Figure_4 2.png]]

