import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seeds for reproducible results
np.random.seed(101)
tf.random.set_seed(101)

# Generate random data for demonstration
rand_a = np.random.uniform(0, 100, (5, 5))
rand_b = np.random.uniform(0, 100, (5, 1))

# Convert numpy arrays to TensorFlow tensors
a_tensor = tf.constant(rand_a, dtype=tf.float32)
b_tensor = tf.constant(rand_b, dtype=tf.float32)

# Perform tensor operations
add_result = a_tensor + b_tensor
mult_result = a_tensor * b_tensor

print("Addition result:")
print(add_result.numpy())
print("\nMultiplication result:")
print(mult_result.numpy())


# Network parameters
n_features = 10
n_dense_neurons = 3
"""
Shape Consistency for Matrix Multiplication
------------------------------------------
x shape: (batch_size, n_features) - batch_size can vary (None)
W shape: (n_features, n_dense_neurons) - must match x's features  
b shape: (n_dense_neurons,) - must match W's output dimension (n_dense_neurons)
"""
# Create input tensor with flexible batch size
x = tf.Variable(tf.zeros([0, n_features]), shape=(None, n_features))

# Initialize weights and biases
W = tf.Variable(tf.random.normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.zeros([n_dense_neurons]))

# Network operations
xW = tf.matmul(x, W)
z = tf.add(xW, b)
a = tf.sigmoid(z)  # Activation function

# Test the layer with random input
test_input = tf.constant(np.random.random([1, n_features]), dtype=tf.float32)
x.assign(test_input)

layer_output = a
print("Layer output:", layer_output.numpy())


# Generate artificial regression data with some noise
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

# Initialize model parameters
m = tf.Variable(0.39)
b = tf.Variable(0.2)

# Training parameters
learning_rate = 0.001
epochs = 100

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Calculate predictions
        y_hat = m * x_data + b
        # Calculate mean squared error
        error = tf.reduce_mean(tf.square(y_label - y_hat))

    # Compute gradients
    gradients = tape.gradient(error, [m, b])

    # Update parameters
    m.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

# Get final parameters
final_slope = m.numpy()
final_intercept = b.numpy()

print(f"Final slope: {final_slope:.4f}")
print(f"Final intercept: {final_intercept:.4f}")


# Generate predictions for plotting
x_test = np.linspace(-1, 11, 10)
y_pred = final_slope * x_test + final_intercept

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_pred, 'r-', label='Regression Line')
plt.plot(x_data, y_label, 'b*', markersize=10, label='Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression Results')
plt.grid(True)
plt.show()
