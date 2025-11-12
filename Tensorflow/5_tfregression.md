## TensorFlow  Regression Implementation

### Data Creation and Visualization

```python
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
```

**Explanation:**
- Creates synthetic linear data with 1 million points for robust training
- `np.linspace()` generates evenly spaced values between 0-10
- Adds Gaussian noise using `np.random.randn()` to simulate real-world data variability
- True relationship follows `y = 0.5x + 5` with added noise
- Combines data into pandas DataFrame for easy manipulation
- Visualizes 250 random samples using matplotlib scatter plot

### Manual TensorFlow Implementation

```python
print("Manual TensorFlow Implementation:")

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
```

**Explanation:**
- `tf.Variable()` creates trainable parameters with initial values (slope=0.5, intercept=1.0)
- Stochastic Gradient Descent optimizer with learning rate 0.001
- Mini-batch training: processes 8 samples at a time for 1000 iterations
- `tf.GradientTape()` automatically tracks operations and computes gradients
- Mean Squared Error loss calculated using `tf.square()` and `tf.reduce_sum()`
- `apply_gradients()` updates variables using computed gradients
- Extracts final parameter values using `.numpy()` method
- Plots regression line (red) over sampled data points


### Data Splitting for Estimator API

```python
# TensorFlow Estimator API
print("\nTensorFlow Estimator API:")

# Split data into train and test sets
x_train, x_eval, y_train, y_eval = train_test_split(
    x_data, y_true, test_size=0.3, random_state=101
)

print(f"Training set size: {x_train.shape}, Evaluation set size: {x_eval.shape}")
```

**Explanation:**
- Uses `train_test_split` from sklearn to split data (70% train, 30% test)
- `random_state=101` ensures reproducible splits
- Prints dataset sizes for verification