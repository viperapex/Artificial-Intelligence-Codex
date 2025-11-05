# Pandas

## Standard Import Syntax
```python
import numpy as np
```

## Elements in sequence

```python
my_list = [1, 2, 3]
print(my_list)
#output: [1, 2, 3]
```

## Casting a list as Array
```python
np.array(my_list)
```


## Check Type
Here nd is n dimensional array
```python
type(np.array(my_list))
#output: <class 'numpy.ndarray'>
```


## Creating Large Arrays within a range

```python
#arange([start,] stop, [, step,],)
a=np.arange(0, 10) #not including 10
print(a)

a = np.arange(0, 11, 2) # with step size of 2
print(a)
#output: [ 0  2  4  6  8 10]
```

## Creating an Array of Zeroes

```python
a = np.zeros((3,5))
print(a)
'''output:
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]'''
```

## Creating Linearly Spaced Arrays

```python
#args:(start, stop, num of elements req evenly spaced)
a = np.linspace(0,11,6)
print(a)
'''output:[ 0.          1.22222222  2.44444444  3.66666667  4.88888889  6.11111111
  7.33333333  8.55555556  9.77777778 11.        ]'''
```


## Creating Random Values

```python
r = np.random.randint(0,10)
print(r)
#output:any value bw 0,9

#for n*n diensional array/matrices
r = np.random.randint(0,100,(3,3))
print(r)
'''output
[[7 2 2]
 [9 1 0]
 [5 0 9]]
'''
```


# NumPy Random Distributions with Code Examples

Here are the common distributions available in NumPy with their corresponding functions and usage examples:

## 1. **Uniform Distribution**
Generates numbers uniformly distributed over a specified range.
```python
import numpy as np
# Generate 10 random numbers between 0 and 1
uniform_data = np.random.uniform(0, 1, 10)
# Generate 5x5 matrix between -2 and 2
uniform_matrix = np.random.uniform(-2, 2, (5, 5))
```

## 2. **Gaussian (Normal) Distribution**
Generates numbers from a normal distribution.
```python
# Generate 10 numbers with mean=0, std=1
normal_data = np.random.normal(0, 1, 10)
# Generate numbers with mean=5, std=2
custom_normal = np.random.normal(5, 2, 100)
```

## 3. **Exponential Distribution**
Models the time between events in a Poisson process.
```python
# Generate 10 numbers with scale=1.0 (lambda=1)
exponential_data = np.random.exponential(1.0, 10)
# With different scale parameter
custom_exp = np.random.exponential(2.5, 100)
```

## 4. **Binomial Distribution**
Represents the number of successes in fixed Bernoulli trials.
```python
# 10 experiments, each with 20 trials and p=0.5 success probability
binomial_data = np.random.binomial(20, 0.5, 10)
# Coin flip simulation: 100 experiments of 10 flips each
coin_flips = np.random.binomial(10, 0.5, 100)
```

## 5. **Poisson Distribution**
Models number of events in fixed intervals.
```python
# Generate 10 numbers with lambda=3 (average rate)
poisson_data = np.random.poisson(3, 10)
# Higher lambda value
high_poisson = np.random.poisson(10, 100)
```

## 6. **Gamma Distribution**
Two-parameter family of continuous distributions.
```python
# Shape=2, scale=2, size=10
gamma_data = np.random.gamma(2, 2, 10)
# Different parameters
custom_gamma = np.random.gamma(1.5, 1, 100)
```

## 7. **Beta Distribution**
Useful for variables limited to [0,1] interval.
```python
# Alpha=2, Beta=5, size=10
beta_data = np.random.beta(2, 5, 10)
# Symmetric beta distribution
symmetric_beta = np.random.beta(0.5, 0.5, 100)
```

## 8. **Chi-Squared Distribution**
Used in hypothesis testing and confidence intervals.
```python
# df=2 degrees of freedom, size=10
chisquare_data = np.random.chisquare(2, 10)
# Higher degrees of freedom
high_chisquare = np.random.chisquare(5, 100)
```

## 9. **Log-Normal Distribution**
Variable whose logarithm is normally distributed.
```python
# Mean=0, sigma=1 for underlying normal distribution
lognormal_data = np.random.lognormal(0, 1, 10)
# Custom parameters
custom_lognormal = np.random.lognormal(1, 0.5, 100)
```

## Additional Useful Distributions:

### Standard Normal
```python
# Standard normal (mean=0, std=1)
std_normal = np.random.standard_normal(100)
```

### Random Integers
```python
# Random integers between low and high (exclusive of high)
integers = np.random.randint(0, 10, 20)
```

### Choice from Array
```python
# Random choice from given array
choices = np.random.choice([1, 2, 3, 4, 5], size=10)
# With probabilities
weighted_choice = np.random.choice(['A', 'B', 'C'], size=10, p=[0.1, 0.3, 0.6])
```

### Setting Random Seed
```python
# For reproducible results
np.random.seed(42)
consistent_data = np.random.normal(0, 1, 10)
integers = np.random.randint(0, 10, 20)
'''
both are linked to the seed defined (42)
'''
```

All these functions return numpy arrays that you can use for statistical analysis, simulations, or machine learning applications.
### Summary of Distributions

| Distribution Type | Description                                       |
| ----------------- | ------------------------------------------------- |
| **Uniform**       | Equal probability across a range                  |
| **Exponential**   | Time between events in a Poisson process          |
| **Binomial**      | Number of successes in a fixed number of trials   |
| **Poisson**       | Number of events in a fixed interval              |
| **Gamma**         | Continuous probability distribution               |
| **Beta**          | Random variables limited to finite intervals      |
| **Chi-Squared**   | Used in hypothesis testing                        |
| **Log-Normal**    | Logarithm of the variable is normally distributed |
|                   |                                                   |

## Find max value of an array

```python
np.random.seed(101)
arr = np.randiint(0,100,10)
#output: [95 11 81 70 63 87 75  9 77 40]
arr.max()
#output: 95
```

## Find min value of an array
```python
arr.min()
#output:9
```

## Find mean value of an array

```python
arr.mean()
#output:
```

## Find index location of max value

```python
arr.argmax()
#output:
```

## Find index location of min value

```python
arr.argmin()
```

## Reshape an Array

```python
print(arr)
#output: [95 11 81 70 63 87 75  9 77 40]
rs=arr.reshape(2, 5)
print(rs)
'''
Output:
[[95 11 81 70 63]
 [87 75  9 77 40]]
'''
```


## Indexing

```python
mat = np.arange(0,100).reshape(10,10)
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

mat[4,3]
#Output: 43

```

## Slicing

```python
mat[:,0] #All rows, 1st column
'''
Output: [ 0 10 20 30 40 50 60 70 80 90]
'''
mat[5, :] #5th row, All columns
'''
Output: [50 51 52 53 54 55 56 57 58 59]
'''
mat[0:3,0:3] #0-3(not including3) R,C
'''
Output: 
[[ 0  1  2]
 [10 11 12]
 [20 21 22]]
'''
```

## Masking
In **NumPy**, **masking** refers to the technique of selectively ignoring or filtering out certain elements in an array based on a condition. This is particularly useful when dealing with datasets that may contain missing or invalid entries. The **numpy.ma** module provides support for masked arrays, which are arrays that can have missing or invalid entries.

### Key Concepts of Masking in NumPy:

1. **Masked Arrays**:
    
    - A masked array is a combination of a standard `numpy.ndarray` and a mask.
    - The mask is an array of boolean values (`True` or `False`), where `True` indicates that the corresponding element in the data array should be ignored or treated as missing.
2. **Creating a Mask**:
    
    - You can create a mask based on conditions applied to the data. For example, if you want to mask all negative values in an array, you can create a boolean mask that is `True` for negative values.
3. **Using Masks**:
    
    - Once you have a mask, you can use it to perform operations on the masked array, such as calculations or visualizations, while ignoring the masked (invalid) entries.
```python
# Create a NumPy array
data = np.array([1, 2, -3, 4, -5, 6])

# Create a mask for negative values
mask = data < 0

# Create a masked array
masked_array = np.ma.masked_array(data, mask)

print("Original Data:", data)
print("Mask:", mask)
print("Masked Array:", masked_array)
'''
Output:
Original Data: [ 1  2 -3  4 -5  6]
Mask: [False False  True False  True False]
Masked Array: [1 2 -- 4 -- 6]
'''

mat > 50
'''
Output:
[[False False False False False False False False False False]
 [False False False False False False False False False False]
 [False False False False False False False False False False]
 [False False False False False False False False False False]
 [False False False False False False False False False False]
 [False  True  True  True  True  True  True  True  True  True]
 [ True  True  True  True  True  True  True  True  True  True]
 [ True  True  True  True  True  True  True  True  True  True]
 [ True  True  True  True  True  True  True  True  True  True]
 [ True  True  True  True  True  True  True  True  True  True]]
'''
my_filter = mat > 50

mat[my_filter]
'''
Output:
[51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98
 99]
'''
```

- **Purpose**: To handle missing or invalid data in arrays.
- **Functionality**: Allows for operations on arrays while ignoring specified elements.
- **Use Cases**: Commonly used in data analysis, scientific computing, and when working with datasets that may have incomplete information.

