from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = np.random.randint(0, 100, (10, 2))
print(data)

scaler_model = MinMaxScaler()
type(scaler_model)

# allows the model to learn the minimum value and maximunm value for each column
scaler_model.fit(data)

# normalising data
scaler_model.transform(data)

# depending on situation both can be done at once
scaler_model.fit_transform(data)


mydata = np.random.randint(0, 101, (50, 4))

print(mydata)

df = pd.DataFrame(data=mydata, columns=['f1', 'f2', 'f3', 'label'])
print(df)


X = df[['f1', 'f2', 'f3']]
y = df['label']


X_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

print(X_train.shape)
print(x_test.shape)
