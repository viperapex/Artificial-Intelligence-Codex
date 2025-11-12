# TensorFlow 2 Regression Example

# Creating Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split  # Fixed import

# 1 Million Points
x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# y = mx + b + noise_levels
y_true = (0.5 * x_data) + 5 + noise

my_data = pd.concat([
    pd.DataFrame(data=x_data, columns=['X Data']),
    pd.DataFrame(data=y_true, columns=['Y'])
], axis=1)

# Visualize sample of data
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.show()


print("Manual TensorFlow 2 Implementation:")

# Initialize variables
m = tf.Variable(0.5)
b = tf.Variable(1.0)
optimizer = tf.optimizers.SGD(learning_rate=0.001)

batch_size = 8
batches = 1000

# Training loop with GradientTape
for i in range(batches):
    rand_ind = np.random.randint(len(x_data), size=batch_size)
    x_batch = tf.constant(x_data[rand_ind], dtype=tf.float32)
    y_batch = tf.constant(y_true[rand_ind], dtype=tf.float32)

    with tf.GradientTape() as tape:
        y_model = m * x_batch + b
        error = tf.reduce_sum(tf.square(y_batch - y_model))

    gradients = tape.gradient(error, [m, b])
    optimizer.apply_gradients(zip(gradients, [m, b]))

model_m = m.numpy()
model_b = b.numpy()

print(f"Manual model - Slope (m): {model_m:.3f}, Intercept (b): {model_b:.3f}")

# Plot manual results
y_hat = x_data * model_m + model_b
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(x_data, y_hat, 'r', linewidth=3)
plt.title("Manual TensorFlow 2 Regression")
plt.show()

# TensorFlow Estimator API
print("\nTensorFlow Estimator API:")

# Split data into train and test sets
x_train, x_eval, y_train, y_eval = train_test_split(
    x_data, y_true, test_size=0.3, random_state=101
)

print(
    f"Training set size: {x_train.shape}, Evaluation set size: {x_eval.shape}")


# Define feature columns and estimator
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# Define input functions using modern tf.data API


def train_input_fn():
    return tf.data.Dataset.from_tensor_slices(
        ({'x': x_train}, y_train)
    ).shuffle(1000).batch(4).repeat()


def eval_input_fn():
    return tf.data.Dataset.from_tensor_slices(
        ({'x': x_eval}, y_eval)
    ).batch(4)


def predict_input_fn():
    return tf.data.Dataset.from_tensor_slices(
        {'x': np.linspace(0, 10, 10)}
    ).batch(1)


# Train the model
estimator.train(input_fn=train_input_fn, steps=1000)

# Evaluate the model
train_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1000)

print(f"Train metrics: {train_metrics}")
print(f"Eval metrics: {eval_metrics}")

# Make predictions
predictions = list(estimator.predict(input_fn=predict_input_fn))
pred_values = [pred['predictions'][0] for pred in predictions]

print(f"Predictions: {pred_values}")

# Plot estimator results
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(np.linspace(0, 10, 10), pred_values,
         'r', linewidth=3, label='Predictions')
plt.title("TensorFlow Estimator Regression")
plt.legend()
plt.show()

print("Completed successfully!")
