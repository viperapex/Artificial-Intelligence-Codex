Note: Scikit-learn is its own machine learning library for python and doesn't support deep neural networks which tensor flow can do
## Standard Import Syntax
### Importing necessary libraries

```python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
```

## Data Scaling Basics

### Creating Sample Data
```python
# Generate small dataset for scaling demonstration
data = np.random.randint(0, 100, (10, 2))
print("Original Data:")
print(data)
```

### Scaling Methods
`MinMaxScaler` scales to range **0–1 by default**, but you can change the range with `feature_range=(a,b)` if needed.
**Method 1: Separate `fit()` and `transform()`**
```python
#scaler_model = MinMaxScaler(feature_range=(0, 1))
scaler_model = MinMaxScaler()

# Fit: Learn data range (min/max) from the data
scaler_model.fit(data)

# Transform: Apply scaling using learned parameters
scaled_data = scaler_model.transform(data)
print("Scaled Data:")
print(scaled_data)
```

**Method 2: Combined `fit_transform()`**
```python
# Fit and transform in one step
fit_transformed_data = scaler_model.fit_transform(data)
```

#### If you use `fit_transform()` on entire dataset before splitting, test data influences the scaling parameters, causing data leakage and over-optimistic performance.
## When to Use Each Approach

### Use Separate `fit()`/`transform()` When:
- **Preventing data leakage** in train-test splits
- **Applying same scaling** to multiple datasets
- **Production environments** with new incoming data

### Use `fit_transform()` When:
- **Quick data exploration**
- **Single dataset processing**
- **Simpler workflows** where data leakage isn't a concern

## Practical Example with Train-Test Split

### Create Larger Dataset
```python
# Generate dataset for modeling
mydata = np.random.randint(0, 101, (50, 4))
df = pd.DataFrame(data=mydata, columns=['f1', 'f2', 'f3', 'label'])
print("DataFrame:")
print(df.head())
```

### Prepare Features and Target
```python
X = df[['f1', 'f2', 'f3']]  # Features
y = df['label']              # Target variable
```

### Split Data First (Critical Step!)
```python
# Always split before scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")
```

### Proper Scaling After Splitting
```python
# Fit scaler ONLY on training data
scaler = MinMaxScaler()
scaler.fit(X_train)

# Transform both sets using same parameters
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Core

1. **Always split data** before preprocessing
2. **Fit scaler on training data only** 
3. **Use same fitted scaler** to transform both train and test sets
4. **Separate `fit()`/`transform()`** prevents data leakage
5. **`fit_transform()`** is convenient but risky with train-test splits

This approach ensures your model evaluation reflects real-world performance where new data is transformed using parameters learned from historical data only.